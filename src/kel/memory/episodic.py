"""Per-session transcript store — the full history of a session, as
opposed to working memory's budget-constrained active window."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from kel.models.types import Message


class EpisodicStore(Protocol):
    def append(self, session_id: str, message: Message) -> None: ...
    def transcript(self, session_id: str) -> list[Message]: ...


class InMemoryEpisodicStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)

    def append(self, session_id: str, message: Message) -> None:
        self._sessions[session_id].append(message)

    def transcript(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def sessions(self) -> list[str]:
        return list(self._sessions.keys())


class FileEpisodicStore:
    """Append-only JSONL per session, one file per session_id. Durable
    across process restarts; still just a local filesystem — swap for a
    real backend by implementing the same two methods."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.jsonl"

    def append(self, session_id: str, message: Message) -> None:
        with self._path(session_id).open("a", encoding="utf-8") as f:
            f.write(message.model_dump_json() + "\n")

    def transcript(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [Message.model_validate_json(line) for line in f if line.strip()]


class SQLiteEpisodicStore:
    """Durable, single-file, SQL-queryable transcript store — the SQLite
    counterpart to `FileEpisodicStore`'s one-JSONL-file-per-session
    approach, and the same zero-new-dependency pattern `kel.caching`'s
    `SQLiteCache` already uses. Durable across process restarts, and,
    being a single file with SQLite's own locking, usable from multiple
    *processes* without extra plumbing (e.g. several worker processes
    behind a web server, all resuming the same sessions) — something
    `InMemoryEpisodicStore` can never do since it never leaves the
    process. This is not a claim to Redis/Postgres-grade concurrent-write
    throughput (SQLite serializes writers), just a durable option that
    doesn't require running a separate server, closing the specific gap
    between "in-memory only" and "bring your own backend."
    """

    def __init__(self, path: str | Path = "kel_episodic.sqlite"):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kel_episodic ("
            "session_id TEXT NOT NULL, seq INTEGER NOT NULL, message TEXT NOT NULL, "
            "PRIMARY KEY (session_id, seq))"
        )
        self._conn.commit()

    def append(self, session_id: str, message: Message) -> None:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM kel_episodic WHERE session_id = ?", (session_id,)
        ).fetchone()
        seq = row[0]
        self._conn.execute(
            "INSERT INTO kel_episodic (session_id, seq, message) VALUES (?, ?, ?)",
            (session_id, seq, message.model_dump_json()),
        )
        self._conn.commit()

    def transcript(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT message FROM kel_episodic WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [Message.model_validate_json(row[0]) for row in rows]

    def sessions(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT session_id FROM kel_episodic").fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()
