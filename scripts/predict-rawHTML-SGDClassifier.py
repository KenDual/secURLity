import argparse
import re
import sys
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

MODEL_PATH = r"D:\phreshphish\models\SGDClassifier_2.joblib"
LABEL_NAMES = {0: "benign", 1: "phish"}
HTML_EXTS = {".html", ".htm"}


# Normalize HTML (cơ chế này phải được giữ nguyên khi xây dựng API, vì sẽ đảm bảo được đầu vào tương ứng với dataset đã train)
def normalize_html(html: str, max_chars: int) -> str:
    if html is None:
        return ""
    html = str(html)
    html = html[: max_chars * 5]
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"\s+", " ", html)
    html = html.strip()
    return html[:max_chars]


# --- Load artifact ---
artifact       = joblib.load(MODEL_PATH)
vec_params     = artifact["vectorizer_params"]
vectorizer     = HashingVectorizer(**vec_params)
clf            = artifact["clf"]
HTML_MAX_CHARS = artifact["html_max_chars"]


def _predict_batch(htmls: list[str]):
    X = vectorizer.transform(htmls)
    pred_ints = clf.predict(X)
    probas = clf.predict_proba(X)
    results = []
    for pred_int, proba in zip(pred_ints, probas):
        pred_int = int(pred_int)
        pred_label = LABEL_NAMES.get(pred_int, str(pred_int))
        score = {
            LABEL_NAMES.get(int(label), str(label)): float(value)
            for label, value in zip(clf.classes_, proba)
        }
        results.append((pred_label, score))
    return results


def predict_html_file(path: str):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    html = normalize_html(html, HTML_MAX_CHARS)
    return _predict_batch([html])[0]


def predict_html_folder(folder: str, recursive: bool = False):
    """Yield (path, pred_label, score) cho từng file .html/.htm trong folder."""
    folder_p = Path(folder)
    if not folder_p.is_dir():
        raise NotADirectoryError(f"Không phải thư mục: {folder}")

    iterator = folder_p.rglob("*") if recursive else folder_p.iterdir()
    files = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in HTML_EXTS
    )

    if not files:
        return

    for p in files:
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
            html = normalize_html(html, HTML_MAX_CHARS)
            pred_label, score = _predict_batch([html])[0]
            yield p, pred_label, score
        except Exception as e:
            yield p, f"ERROR: {type(e).__name__}: {e}", None


def _print_result(path, pred, score):
    print(f"\nFile      : {path}")
    print(f"Prediction: {pred}")
    print(f"Score     : {score}")


def main():
    ap = argparse.ArgumentParser(
        description="Predict benign/phish cho 1 file HTML hoặc cả 1 folder."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--file", type=str, help="đường dẫn 1 file HTML")
    g.add_argument("--folder", type=str, help="thư mục chứa các file .html/.htm")
    ap.add_argument("--recursive", action="store_true",
                    help="duyệt đệ quy các subfolder (chỉ áp dụng với --folder)")
    ap.add_argument("positional", nargs="?", default=None,
                    help="(tuỳ chọn) đường dẫn file HTML, để tương thích cách gọi cũ")
    args = ap.parse_args()

    if args.folder:
        folder = args.folder
        print(f"Folder    : {folder}  (recursive={args.recursive})")
        n = ok = phish_n = benign_n = err = 0
        for path, pred, score in predict_html_folder(folder, recursive=args.recursive):
            n += 1
            if score is None:
                err += 1
            else:
                ok += 1
                if pred == "phish":
                    phish_n += 1
                elif pred == "benign":
                    benign_n += 1
            _print_result(path, pred, score)
        if n == 0:
            print(f"\n(Không tìm thấy file .html/.htm trong {folder})")
        else:
            print(
                f"\n--- Summary ---\n"
                f"Total   : {n}\n"
                f"benign  : {benign_n}\n"
                f"phish   : {phish_n}\n"
                f"errors  : {err}"
            )
        return

    # Single-file mode (giữ tương thích cách dùng cũ)
    html_path = args.file or args.positional
    if not html_path:
        html_path = input("Nhập đường dẫn file HTML: ")

    pred, score = predict_html_file(html_path)
    _print_result(html_path, pred, score)


if __name__ == "__main__":
    main()
