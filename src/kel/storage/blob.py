"""Content-addressed blob storage (DESIGN.md 3.9): the id is
sha256(content), so identical content is stored once and re-`put`ting the
same bytes is free — this is what "avoids re-computation and enables
replay" means in practice, not a special caching layer bolted on top."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def put(self, data: bytes) -> str: ...
    def get(self, content_id: str) -> bytes: ...
    def exists(self, content_id: str) -> bool: ...
    def delete(self, content_id: str) -> None: ...


class LocalBlobStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, content_id: str) -> Path:
        # git-style sharding by the first two hex chars, so one directory
        # doesn't end up with millions of files in it
        return self.directory / content_id[:2] / content_id

    def put(self, data: bytes) -> str:
        content_id = hashlib.sha256(data).hexdigest()
        path = self._path(content_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return content_id

    def get(self, content_id: str) -> bytes:
        path = self._path(content_id)
        if not path.exists():
            raise KeyError(content_id)
        return path.read_bytes()

    def exists(self, content_id: str) -> bool:
        return self._path(content_id).exists()

    def delete(self, content_id: str) -> None:
        path = self._path(content_id)
        if path.exists():
            path.unlink()
