"""Response cache backends. `InMemoryCache` for process-lifetime caching
(dev/test loops), `SQLiteCache` for a durable cache that survives process
restarts. No Redis adapter (out of scope: needs a running Redis server,
not something kel should assume)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from kel.models.types import ModelResponse


class ResponseCache(Protocol):
    def get(self, key: str) -> ModelResponse | None: ...
    def set(self, key: str, response: ModelResponse) -> None: ...


class InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, ModelResponse] = {}

    def get(self, key: str) -> ModelResponse | None:
        return self._store.get(key)

    def set(self, key: str, response: ModelResponse) -> None:
        self._store[key] = response

    def __len__(self) -> int:
        return len(self._store)


class SQLiteCache:
    def __init__(self, path: str | Path = "kel_cache.sqlite"):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS kel_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> ModelResponse | None:
        row = self._conn.execute("SELECT response FROM kel_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return ModelResponse.model_validate_json(row[0])

    def set(self, key: str, response: ModelResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kel_cache (key, response) VALUES (?, ?)", (key, response.model_dump_json())
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
