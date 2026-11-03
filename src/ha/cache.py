"""SQLite key-value cache for expensive calls (LLM generations, judge calls, reranker pairs).

Keys are sha256 of canonical JSON of whatever determines the result (prompt, model digest,
decoding params ...). Caches produced on Kaggle are merged back with `merge_from`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ha.config import sha256_hex
from ha.utils import utc_now_iso


class KVCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def key_for(obj: Any) -> str:
        return sha256_hex(obj)

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, created) VALUES (?, ?, ?)",
                (key, payload, utc_now_iso()),
            )
            self._conn.commit()

    def get_or_compute(self, key_obj: Any, fn: Callable[[], Any]) -> Any:
        key = self.key_for(key_obj)
        hit = self.get(key)
        if hit is not None:
            return hit
        value = fn()
        self.set(key, value)
        return value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]

    def merge_from(self, other: str | Path) -> int:
        """Copy entries from another cache file; existing keys are kept. Returns rows added."""
        before = len(self)
        with self._lock:
            self._conn.execute("ATTACH DATABASE ? AS other", (str(other),))
            self._conn.execute("INSERT OR IGNORE INTO kv SELECT key, value, created FROM other.kv")
            self._conn.commit()
            self._conn.execute("DETACH DATABASE other")
        return len(self) - before

    def close(self) -> None:
        self._conn.close()
