"""Human-in-the-loop for kel.runtime graphs — the equivalent of
LangGraph's `interrupt()`. A node function raises `Interrupt(payload)` to
pause the graph; `run_graph` catches it and returns a `GraphRun` with
`interrupted=True` instead of propagating the exception. The caller shows
`interrupt_payload` to a human, gets their input, and calls `resume_graph`
with it to continue exactly where execution paused."""

from __future__ import annotations

from typing import Any


class Interrupt(Exception):
    def __init__(self, payload: Any):
        super().__init__(f"graph interrupted: {payload!r}")
        self.payload = payload
