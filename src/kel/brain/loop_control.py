"""Budget-aware continue-vs-stop decision (DESIGN.md 3.14). Honest v1
scope: this is a fixed-reserve threshold check, not real marginal-value
estimation — actually modeling "is another iteration worth it" needs
outcome data (did past extra iterations actually help?) this project
doesn't have yet. It's a placeholder for that, with the same call
signature a smarter version could fill in later."""

from __future__ import annotations

from kel.budget.types import BudgetSnapshot


def should_continue(
    snapshot: BudgetSnapshot, *, min_tokens_reserve: int = 200, min_cost_reserve: float = 0.01
) -> bool:
    if snapshot.tokens_remaining is not None and snapshot.tokens_remaining < min_tokens_reserve:
        return False
    if snapshot.cost_usd_remaining is not None and snapshot.cost_usd_remaining < min_cost_reserve:
        return False
    if snapshot.wall_seconds_remaining is not None and snapshot.wall_seconds_remaining <= 0:
        return False
    return True
