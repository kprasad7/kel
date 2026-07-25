"""Checkpoint = a snapshot of graph state after one node's execution
(DESIGN.md 3.10: "every node transition persists state so a run can pause/
resume/branch from any point"). A plain dataclass, not a pydantic model —
state can legitimately hold arbitrary Python objects (Message, ModelResponse,
...), not just JSON-safe values, and forcing that would make graph state
awkward to work with for no v1 benefit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Checkpoint:
    run_id: str
    step: int
    node: str
    state: dict[str, Any]


class CheckpointStore(Protocol):
    def save(self, checkpoint: Checkpoint) -> None: ...
    def load_latest(self, run_id: str) -> Checkpoint | None: ...
    def history(self, run_id: str) -> list[Checkpoint]: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._store: dict[str, list[Checkpoint]] = defaultdict(list)

    def save(self, checkpoint: Checkpoint) -> None:
        self._store[checkpoint.run_id].append(checkpoint)

    def load_latest(self, run_id: str) -> Checkpoint | None:
        items = self._store.get(run_id)
        return items[-1] if items else None

    def history(self, run_id: str) -> list[Checkpoint]:
        return list(self._store.get(run_id, []))
