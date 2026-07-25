"""A Cassette is a recorded sequence of model interactions
(DESIGN.md 3.7: "capture real model + tool responses, replay
deterministically in CI"). Plain JSON on disk — readable, diffable in a
PR, not an opaque binary format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kel.models.types import ModelResponse


class Interaction(BaseModel):
    request: dict[str, Any]
    response: ModelResponse


class Cassette(BaseModel):
    interactions: list[Interaction] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
