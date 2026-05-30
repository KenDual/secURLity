import os
from pathlib import Path

_APP_DIR = Path(__file__).parent        # .../webapp/app/
_WEBAPP_ROOT = _APP_DIR.parent          # .../webapp/

MODEL_DIR = _WEBAPP_ROOT / "bundled_models"
SCRIPTS_DIR = _APP_DIR / "scripts"

# ── Ensemble ──────────────────────────────────────────────────────────────────
# USER: edit ENSEMBLE_THRESHOLD to tune detection sensitivity.
#   0.5  = balanced (default)
#   0.7  = more conservative (fewer false positives, may miss more malicious URLs)
#   0.3  = more aggressive (catches more, but more false positives)
ENSEMBLE_THRESHOLD: float = 0.5

# ── SGD / HTML fetch ──────────────────────────────────────────────────────────
SGD_HTML_FETCH_TIMEOUT: float = 10.0   # seconds
SGD_MAX_HTML_BYTES: int = 2_000_000    # bytes read before truncation

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT = "10/minute"

# ── Database ──────────────────────────────────────────────────────────────────
_db_env = os.environ.get("DB_PATH")
if _db_env:
    DB_PATH = Path(_db_env)
elif Path("/data").exists():
    DB_PATH = Path("/data/scans.db")    # HF Spaces persistent volume
else:
    DB_PATH = Path("scans.db")          # local dev fallback
