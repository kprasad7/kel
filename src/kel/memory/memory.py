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
    memory" so callers don't have to remember to do both.

    **Recalls prior turns for a session automatically.** If `working` isn't
    given explicitly, working memory is seeded from `episodic`'s existing
    transcript for `session_id` (if any) instead of always starting empty.
    This matters whenever the caller (an `Agent`, or code built on one) is
    reconstructed fresh but `episodic` is a store that outlives that —
    e.g. a `FileEpisodicStore` shared across process restarts, or across a
    Streamlit-style script rerun per user interaction: pass the same
    `episodic` instance and `session_id` again and the conversation
    resumes instead of restarting silently. An `InMemoryEpisodicStore`
    (the default) doesn't outlive the process, so this only has an effect
    when you supply a durable store yourself.
    """

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
        self.episodic = episodic or InMemoryEpisodicStore()
        if working is not None:
            self.working = working
        else:
            self.working = WorkingMemory(messages=self.episodic.transcript(session_id), max_tokens=8000)
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural

    def remember_turn(self, message: Message) -> None:
        self.working.add(message)
        self.episodic.append(self.session_id, message)
