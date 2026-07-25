"""Approximate USD-per-million-token pricing, used only to give
BudgetTracker a cost estimate. Deliberately a small hardcoded table, not a
live-pricing integration — update as needed; unknown models cost $0 (with a
tracked `unpriced` flag) rather than guessing."""

from __future__ import annotations

from kel.models.types import Usage

# (input $/1M tokens, output $/1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-4-8": (15.0, 75.0),
    "anthropic:claude-sonnet-5": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (0.80, 4.0),
    "openai:gpt-5.2": (5.0, 15.0),
    "openai:gpt-5.2-mini": (0.25, 2.0),
}


def estimate_cost_usd(provider: str, model_id: str, usage: Usage) -> float:
    rates = _PRICING.get(f"{provider}:{model_id}")
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return (usage.input_tokens / 1_000_000) * input_rate + (usage.output_tokens / 1_000_000) * output_rate


def is_priced(provider: str, model_id: str) -> bool:
    return f"{provider}:{model_id}" in _PRICING
