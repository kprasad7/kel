"""Token estimation. v1 uses a documented approximation (chars/4) rather
than a real tokenizer — pulling in tiktoken/provider-specific tokenizers is
real work with real accuracy tradeoffs per model family; this is a known
gap, not a hidden one. Swap `estimate_tokens` for a real tokenizer via the
`counter` param on ContextWindow when accuracy matters more than zero
dependencies."""

from __future__ import annotations

from kel.models.types import Message


def estimate_tokens(text: str) -> int:
    """~4 chars/token is the commonly-cited rule of thumb for English text
    across GPT/Claude-family tokenizers. Good enough for budget headroom
    decisions, not for exact billing (use kel.budget's real usage figures
    for that — this is only ever an upper-bound-ish estimate pre-call)."""
    return max(1, len(text) // 4) if text else 0


def estimate_message_tokens(message: Message) -> int:
    return estimate_tokens(message.text) + 4  # small per-message overhead, matches common estimation conventions


def estimate_total_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)
