from __future__ import annotations

from kel.memory.episodic import EpisodicStore, InMemoryEpisodicStore
from kel.memory.procedural import ProceduralMemory
from kel.memory.semantic import SemanticMemory
from kel.memory.working import WorkingMemory
from kel.models.types import Message


class Memory:
    """Convenience facade over the four memory layers (DESIGN.md 3.3). Each
    layer is independently usable/testable on its own — this just wires
    the common case of "append a turn to both working and episodic
    memory" so callers don't have to remember to do both."""

    def __init__(
        self,
        *,
        session_id: str = "default",
        working: WorkingMemory | None = None,
        episodic: EpisodicStore | None = None,
        semantic: SemanticMemory | None = None,
        procedural: ProceduralMemory | None = None,
    ):
        self.session_id = session_id
        self.working = working or WorkingMemory(max_tokens=8000)
        self.episodic = episodic or InMemoryEpisodicStore()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural

    def remember_turn(self, message: Message) -> None:
        self.working.add(message)
        self.episodic.append(self.session_id, message)
