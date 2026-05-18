import os
import gc
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import polars as pl
import joblib
import sklearn
import scipy
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import vstack as sp_vstack

# Config
DATA_DIR  = Path(r"D:\phreshphish\data")
MODEL_DIR = Path(r"D:\phreshphish\models")
CACHE_DIR = Path(r"D:\phreshphish\cache")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DOCS = int(os.getenv("CHUNK_DOCS", "1000"))

HTML_MAX_CHARS    = int(os.getenv("HTML_MAX_CHARS",   "10000"))
N_FEATURES        = int(os.getenv("N_FEATURES",       str(2**20)))
NGRAM_RANGE       = (3, 4)
EPOCHS            = int(os.getenv("EPOCHS",           "1"))
PREFETCH_WORKERS  = max(1, int(os.getenv("PREFETCH_WORKERS", "4")))
PREFETCH_DEPTH    = max(1, int(os.getenv("PREFETCH_DEPTH",   "2")))
USE_CACHE         = os.getenv("USE_CACHE", "1") == "1"

CLASSES = np.array(["benign", "phish"])

VEC_PARAMS = dict(
    analyzer="char",
    ngram_range=NGRAM_RANGE,
    n_features=N_FEATURES,
    lowercase=False,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32,
)

CACHE_CONFIG = {
    "n_features":     N_FEATURES,
    "ngram_range":    list(NGRAM_RANGE),
    "analyzer":       "char",
    "lowercase":      False,
    "norm":           "l2",
    "alternate_sign": False,
    "html_max_chars": HTML_MAX_CHARS,
    "preprocess":     "normalize_html_v1",
}
CACHE_KEY    = hashlib.md5(json.dumps(CACHE_CONFIG, sort_keys=True).encode()).hexdigest()[:12]
CACHE_SUBDIR = CACHE_DIR / CACHE_KEY
CACHE_SUBDIR.mkdir(parents=True, exist_ok=True)


# Normalize HTML thành các dòng đơn giản
def _normalize_html(series: pl.Series) -> pl.Series:
    """Polars regex normalize. Slice trước 1 lần để bound regex work."""
    return (
        series.fill_null("")
              .str.slice(0, HTML_MAX_CHARS * 5)
              .str.replace_all(r">\s+<", "><")
              .str.replace_all(r"\s+", " ")
              .str.strip_chars()
              .str.slice(0, HTML_MAX_CHARS)
    )


def vectorize_file(file_path: str, cache_dir: str, use_cache: bool) -> str:
    """
    Compute -> ghi cache xuống disk -> trả về string path.
    KHÔNG return (X, y) trực tiếp để tránh pickle qua IPC.
    Main process sẽ joblib.load(path) sau khi nhận path.
    """
    f          = Path(file_path)
    cache_dir  = Path(cache_dir)
    cache_path = cache_dir / f"{f.stem}.joblib"

    if use_cache and cache_path.exists():
        return str(cache_path)

    df    = pl.read_parquet(f, columns=["html", "label"])
    htmls = _normalize_html(df["html"]).to_list()
    # Convert label strings -> int8 để cache nhẹ và pickle nhanh hơn
    label_list = df["label"].to_list()
    y = np.fromiter(
        (1 if v == "phish" else 0 for v in label_list),
        dtype=np.int8,
        count=len(label_list),
    )
    del df, label_list

    vec = HashingVectorizer(**VEC_PARAMS)
    parts = []
    for i in range(0, len(htmls), CHUNK_DOCS):
        parts.append(vec.transform(htmls[i:i + CHUNK_DOCS]))
    X = sp_vstack(parts).tocsr() if len(parts) > 1 else parts[0]
    del htmls, parts
    gc.collect()

    joblib.dump((X, y), cache_path, compress=3)
    del X, y
    gc.collect()

    return str(cache_path)

def parallel_batches(file_list, executor, use_cache, desc):
    futures  = []
    next_idx = 0

    while next_idx < min(PREFETCH_DEPTH, len(file_list)):
        futures.append(executor.submit(
            vectorize_file, str(file_list[next_idx]), str(CACHE_SUBDIR), use_cache
        ))
        next_idx += 1

    pbar = tqdm(total=len(file_list), desc=desc)
    for _ in range(len(file_list)):
        cache_path = futures.pop(0).result()

        if next_idx < len(file_list):
            futures.append(executor.submit(
                vectorize_file, str(file_list[next_idx]), str(CACHE_SUBDIR), use_cache
            ))
            next_idx += 1

        X, y = joblib.load(cache_path)

        if y.dtype == object or y.dtype.kind in ("U", "S"):
            y = np.fromiter(
                (1 if v == "phish" else 0 for v in y),
                dtype=np.int8,
                count=len(y),
            )

        yield X, y
        del X, y
        pbar.update(1)
    pbar.close()    
    

# Main
def main():
    train_files = sorted(DATA_DIR.glob("train-*.parquet"))
    test_files  = sorted(DATA_DIR.glob("test-*.parquet"))

    if not train_files:
        raise FileNotFoundError(f"Không tìm thấy train-*.parquet trong {DATA_DIR}")
    if not test_files:
        raise FileNotFoundError(f"Không tìm thấy test-*.parquet trong {DATA_DIR}")

    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")
    print(
        f"Config: n_features=2^{int(np.log2(N_FEATURES))} ({N_FEATURES}), "
        f"ngram={NGRAM_RANGE}, html_max_chars={HTML_MAX_CHARS}, "
        f"epochs={EPOCHS}, workers={PREFETCH_WORKERS}, depth={PREFETCH_DEPTH}"
    )
    print(f"Cache: {'ON' if USE_CACHE else 'OFF'} | key={CACHE_KEY} | dir={CACHE_SUBDIR}")

    # --- Class weights (sample 5 file đầu) ---
    print("\nEstimating class distribution...")
    sample_labels = []
    for f in train_files[: min(5, len(train_files))]:
        s = pl.read_parquet(f, columns=["label"])["label"].to_list()
        sample_labels.extend(1 if v == "phish" else 0 for v in s)
    sample_labels = np.array(sample_labels, dtype=np.int8)

    CLASSES_INT = np.array([0, 1], dtype=np.int8)
    weights = compute_class_weight("balanced", classes=CLASSES_INT, y=sample_labels)
    class_weight = dict(zip(CLASSES_INT.tolist(), weights))
    print(f"Class weights (0=benign, 1=phish): {class_weight}")

    # --- SGD ---
    clf = SGDClassifier(
        loss="log_loss",
        class_weight=class_weight,
        max_iter=1,
        tol=None,
        random_state=42,
        average=False,
    )

    # ====================
    # Train
    # ====================
    print("\n=== Training ===")
    t_train = time.perf_counter()
    gc.disable()
    try:
        for epoch in range(EPOCHS):
            with ProcessPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
                for X, y in parallel_batches(
                    train_files, ex, USE_CACHE, desc=f"Epoch {epoch+1}/{EPOCHS}"
                ):
                    clf.partial_fit(X, y, classes=CLASSES_INT)
                    del X, y
    finally:
        gc.enable()
    train_time = time.perf_counter() - t_train
    print(f"Training time: {train_time:.1f}s  ({train_time/60:.2f} phút)")

    # ====================
    # Save model
    # ====================
    artifact = {
        "vectorizer_params":     VEC_PARAMS,
        "clf":                   clf,
        "classes":               CLASSES,
        "html_max_chars":        HTML_MAX_CHARS,
        "n_features":            N_FEATURES,
        "ngram_range":           NGRAM_RANGE,
        "preprocess":            "normalize_html_v1",
        "preprocess_description":"polars: slice 5x -> replace >\\s+< -> ><, \\s+ -> ' ', strip, slice",
        "cache_key":             CACHE_KEY,
        "train_time_sec":        train_time,
        "sklearn_version":       sklearn.__version__,
        "scipy_version":         scipy.__version__,
        "numpy_version":         np.__version__,
        "polars_version":        pl.__version__,
    }
    model_path = MODEL_DIR / "SGD_hashing_v3.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved model -> {model_path}")

    # ====================
    # Evaluate
    # ====================
    print("\n=== Evaluating ===")
    t_eval = time.perf_counter()

    all_y_true, all_y_pred, all_y_score = [], [], []

    with ProcessPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
        for X, y in parallel_batches(test_files, ex, USE_CACHE, desc="Test"):
            preds = clf.predict(X)
            try:
                scores = clf.decision_function(X)
            except Exception:
                scores = None

            all_y_true.extend(y.tolist())
            all_y_pred.extend(preds.tolist())
            if scores is not None:
                all_y_score.extend(np.asarray(scores).tolist())
            del X, y

    eval_time = time.perf_counter() - t_eval
    print(f"Eval time: {eval_time:.1f}s")

    print("\n--- Classification report ---")
    print(classification_report(
        all_y_true, all_y_pred, digits=4,
        target_names=["benign", "phish"],
        labels=[0, 1],
    ))

    if all_y_score:
        try:
            auc = roc_auc_score(all_y_true, all_y_score)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"AUC compute failed: {e}")

if __name__ == "__main__":
    main()