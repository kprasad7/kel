"""Bounded agentic loop primitive (DESIGN.md 3.2): explicit exit
conditions, a step budget, and stuck-loop detection via a repeated-action
heuristic — the thing LangChain's AgentExecutor notoriously lacks."""

from __future__ import annotations

import time

from kel.context.errors import LoopBudgetExceededError, StuckLoopError


class Loop:
    def __init__(
        self,
        *,
        max_iterations: int = 25,
        max_wall_seconds: float | None = None,
        stuck_window: int = 4,
        stuck_threshold: int = 3,
    ):
        self.max_iterations = max_iterations
        self.max_wall_seconds = max_wall_seconds
        self.stuck_window = stuck_window
        self.stuck_threshold = stuck_threshold
        self.iteration = 0
        self._start = time.monotonic()
        self._recent_signatures: list[str] = []

    @property
    def wall_seconds_used(self) -> float:
        return time.monotonic() - self._start

    def step(self) -> int:
        """Call at the top of every loop iteration. Raises
        LoopBudgetExceededError if the loop has run out of iterations or
        time. Returns the new iteration count on success."""
        self.iteration += 1
        if self.iteration > self.max_iterations:
            raise LoopBudgetExceededError(f"loop exceeded max_iterations={self.max_iterations}")
        if self.max_wall_seconds is not None and self.wall_seconds_used > self.max_wall_seconds:
            raise LoopBudgetExceededError(f"loop exceeded max_wall_seconds={self.max_wall_seconds}")
        return self.iteration

    def record_action(self, signature: str) -> None:
        """Feed a signature identifying the action just taken (e.g. a tool
        name + a hash of its arguments). Raises StuckLoopError if the same
        signature dominates the recent window — a repeated-tool-call /
        no-progress heuristic, not a semantic understanding of "stuck"."""
        self._recent_signatures.append(signature)
        window = self._recent_signatures[-self.stuck_window :]
        if len(window) < self.stuck_threshold:
            return
        repeat_count = window.count(signature)
        if repeat_count >= self.stuck_threshold:
            raise StuckLoopError(
                f"loop appears stuck: action {signature!r} repeated {repeat_count} times "
                f"in the last {len(window)} iterations",
                signature=signature,
                repeat_count=repeat_count,
            )

    @property
    def exhausted(self) -> bool:
        return self.iteration >= self.max_iterations
