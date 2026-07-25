"""Episodic -> semantic consolidation (DESIGN.md 3.3). Deliberately a plain
function, not a scheduler — v1 gives you the hook; wiring it to a cron/
trigger is an application concern, not something kel.memory should own."""

from __future__ import annotations

from collections.abc import Callable

from kel.memory.episodic import EpisodicStore
from kel.memory.semantic import SemanticMemory
from kel.models.types import Message


def consolidate(
    episodic: EpisodicStore,
    session_id: str,
    semantic: SemanticMemory,
    *,
    summarize: Callable[[list[Message]], str],
) -> str | None:
    """Summarize a session's full transcript via `summarize` (typically an
    LLM call) and store the result as a semantic fact tagged with the
    session it came from. Returns the new fact's id, or None if the
    session had nothing to consolidate."""
    transcript = episodic.transcript(session_id)
    if not transcript:
        return None
    summary_text = summarize(transcript)
    return semantic.remember(summary_text, metadata={"session_id": session_id, "source": "consolidation"})
