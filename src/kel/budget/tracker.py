from __future__ import annotations

import time

from kel.budget.errors import BudgetExceededError
from kel.budget.types import Budget, BudgetSnapshot
from kel.models.types import Usage


class BudgetTracker:
    """Mutable accumulator against a fixed Budget. One tracker per
    session/agent/graph-run — share it across every call that should count
    against the same limit (DESIGN.md 3.8: "enforcement points at model
    call, loop iteration, and subagent spawn boundaries")."""

    def __init__(self, budget: Budget | None = None):
        self.budget = budget or Budget()
        self.tokens_used = 0
        self.cost_usd_used = 0.0
        self.tool_calls_used = 0
        self._start = time.monotonic()

    @property
    def wall_seconds_used(self) -> float:
        return time.monotonic() - self._start

    def snapshot(self) -> BudgetSnapshot:
        b = self.budget
        return BudgetSnapshot(
            tokens_used=self.tokens_used,
            cost_usd_used=self.cost_usd_used,
            tool_calls_used=self.tool_calls_used,
            wall_seconds_used=self.wall_seconds_used,
            tokens_remaining=None if b.max_tokens is None else b.max_tokens - self.tokens_used,
            cost_usd_remaining=None if b.max_cost_usd is None else b.max_cost_usd - self.cost_usd_used,
            tool_calls_remaining=None if b.max_tool_calls is None else b.max_tool_calls - self.tool_calls_used,
            wall_seconds_remaining=(
                None if b.max_wall_seconds is None else b.max_wall_seconds - self.wall_seconds_used
            ),
        )

    def check_wall_time(self) -> None:
        b = self.budget
        if b.max_wall_seconds is not None and self.wall_seconds_used > b.max_wall_seconds:
            raise BudgetExceededError(
                f"wall time budget exceeded: {self.wall_seconds_used:.1f}s > {b.max_wall_seconds}s",
                dimension="wall_seconds",
            )

    def record_usage(self, usage: Usage, cost_usd: float = 0.0) -> None:
        self.tokens_used += usage.total_tokens
        self.cost_usd_used += cost_usd
        self._enforce()

    def record_tool_call(self) -> None:
        self.tool_calls_used += 1
        self._enforce()

    def _enforce(self) -> None:
        b = self.budget
        self.check_wall_time()
        if b.max_tokens is not None and self.tokens_used > b.max_tokens:
            raise BudgetExceededError(
                f"token budget exceeded: {self.tokens_used} > {b.max_tokens}", dimension="tokens"
            )
        if b.max_cost_usd is not None and self.cost_usd_used > b.max_cost_usd:
            raise BudgetExceededError(
                f"cost budget exceeded: ${self.cost_usd_used:.4f} > ${b.max_cost_usd}", dimension="cost_usd"
            )
        if b.max_tool_calls is not None and self.tool_calls_used > b.max_tool_calls:
            raise BudgetExceededError(
                f"tool call budget exceeded: {self.tool_calls_used} > {b.max_tool_calls}",
                dimension="tool_calls",
            )
