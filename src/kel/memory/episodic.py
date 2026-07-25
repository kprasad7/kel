"""Per-session transcript store — the full history of a session, as
opposed to working memory's budget-constrained active window."""

from __future__ import annotations

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
