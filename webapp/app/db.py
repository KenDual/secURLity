import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from typing import Optional

import aiosqlite

from .config import DB_PATH

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash         TEXT    NOT NULL,
    url_display      TEXT    NOT NULL,
    cnn_prob         REAL,
    xgb_prob         REAL,
    ensemble_prob    REAL,
    ensemble_verdict TEXT,
    sgd_enabled      INTEGER NOT NULL DEFAULT 0,
    sgd_prob         REAL,
    latency_ms       INTEGER,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_created_at ON scans(created_at DESC);
"""


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(_CREATE_SQL)
        await db.commit()


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def display_url(url: str) -> str:
    """Strip query string; keep scheme://host + path[:40]."""
    try:
        p = urlsplit(url)
        base = f"{p.scheme}://{p.netloc}" if p.scheme else p.netloc
        path = p.path[:40] + ("…" if len(p.path) > 40 else "")
        return (base + path)[:120]
    except Exception:
        return url[:120]


async def insert_scan(
    *,
    url_hash: str,
    url_display: str,
    cnn_prob: Optional[float],
    xgb_prob: Optional[float],
    ensemble_prob: Optional[float],
    ensemble_verdict: Optional[str],
    sgd_enabled: int,
    sgd_prob: Optional[float],
    latency_ms: Optional[int],
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO scans
               (url_hash, url_display, cnn_prob, xgb_prob, ensemble_prob,
                ensemble_verdict, sgd_enabled, sgd_prob, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (url_hash, url_display, cnn_prob, xgb_prob, ensemble_prob,
             ensemble_verdict, sgd_enabled, sgd_prob, latency_ms),
        )
        await db.commit()


async def get_recent(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?",
            (min(limit, 100),),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
