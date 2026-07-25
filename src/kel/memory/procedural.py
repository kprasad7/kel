"""Learned prompt/tool patterns, stored as .md files (DESIGN.md 3.3 + 2.4:
".md files are the prompt/config source of truth"). This is where
kel.heal's (failure -> diagnosis -> fix) records and kel.brain's routing
patterns are meant to accumulate over time — a plain directory of
human-readable files, not an opaque store."""

from __future__ import annotations

from pathlib import Path


class ProceduralMemory:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, content: str) -> None:
        (self.directory / f"{name}.md").write_text(content, encoding="utf-8")

    def load(self, name: str) -> str | None:
        path = self.directory / f"{name}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.md"))

    def delete(self, name: str) -> None:
        path = self.directory / f"{name}.md"
        if path.exists():
            path.unlink()
