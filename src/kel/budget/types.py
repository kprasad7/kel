from __future__ import annotations

from pydantic import BaseModel


class Budget(BaseModel):
    """Threaded through a call path (DESIGN.md 3.8). v1 enforcement policy is
    "raise" only — truncate/summarize policies are a kel.context concern
    (eviction), not something budget silently does on your behalf."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_tool_calls: int | None = None
    max_wall_seconds: float | None = None


class BudgetSnapshot(BaseModel):
    """Point-in-time read of a BudgetTracker, safe to log/trace."""

    tokens_used: int
    cost_usd_used: float
    tool_calls_used: int
    wall_seconds_used: float
    tokens_remaining: int | None
    cost_usd_remaining: float | None
    tool_calls_remaining: int | None
    wall_seconds_remaining: float | None
