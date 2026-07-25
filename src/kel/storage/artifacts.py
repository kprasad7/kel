"""Human-friendly name -> content-addressed blob, for artifacts you want
to look up by a stable name (a trace export, a media blob) rather than by
hash. The manifest is a plain JSON file, readable without kel installed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kel.storage.blob import BlobStore


class ArtifactStore:
    def __init__(self, blob_store: BlobStore, directory: str | Path):
        self.blob_store = blob_store
        self.manifest_path = Path(directory) / "manifest.json"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, dict[str, str]] = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict[str, str]]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

    def save(self, name: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        content_id = self.blob_store.put(data)
        self._manifest[name] = {"content_id": content_id, "content_type": content_type}
        self._save_manifest()
        return content_id

    def load(self, name: str) -> bytes:
        entry = self._manifest.get(name)
        if entry is None:
            raise KeyError(name)
        return self.blob_store.get(entry["content_id"])

    def content_type(self, name: str) -> str:
        entry = self._manifest.get(name)
        if entry is None:
            raise KeyError(name)
        return entry["content_type"]

    def list(self) -> list[str]:
        return sorted(self._manifest)
