"""Working memory = the current session's active context. Deliberately
just a thin, semantically-named subclass of ContextWindow (DESIGN.md 3.2)
rather than a separate implementation — the "layered memory" story only
makes sense if working memory actually is the context window, not a
second copy of the same state that can drift from it."""

from __future__ import annotations

from kel.context.window import ContextWindow


class WorkingMemory(ContextWindow):
    pass
