"""Pluggable eviction policies: `(messages, max_tokens) -> messages` that
best-effort bring the list under budget. ContextWindow calls one of these
whenever an append pushes it over max_tokens (DESIGN.md 3.2)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from kel.context.tokens import estimate_message_tokens, estimate_total_tokens
from kel.models.types import Message

EvictionPolicy = Callable[[list["Message"], int], list["Message"]]


def sliding_window_eviction(messages: list[Message], max_tokens: int) -> list[Message]:
    """Drop oldest messages until under budget or only one message remains.

    Uses a deque (O(1) pop-from-front, vs. O(n) for `list.pop(0)`) and
    decrements a running token total instead of resumming the whole list
    on every pop — evicting k of n messages is O(n) total, not O(k*n).
    """
    kept = deque(messages)
    total = estimate_total_tokens(messages)
    while total > max_tokens and len(kept) > 1:
        removed = kept.popleft()
        total -= estimate_message_tokens(removed)
    return list(kept)


def make_summarization_eviction(
    summarize: Callable[[list[Message]], Message], *, keep_recent: int = 4
) -> EvictionPolicy:
    """Summarize everything except the most recent `keep_recent` messages
    into a single message via `summarize` (typically an LLM call), falling
    back to sliding-window eviction on the result if it's still over
    budget (e.g. the summary itself came back long)."""

    def policy(messages: list[Message], max_tokens: int) -> list[Message]:
        if estimate_total_tokens(messages) <= max_tokens or len(messages) <= keep_recent:
            return sliding_window_eviction(messages, max_tokens)
        to_summarize, recent = messages[:-keep_recent], messages[-keep_recent:]
        summary_message = summarize(to_summarize)
        summary_tokens = estimate_message_tokens(summary_message)

        if summary_tokens >= max_tokens:
            # the summary alone doesn't fit; no way to protect it, so fall
            # back to plain sliding-window eviction over everything
            return sliding_window_eviction([summary_message, *recent], max_tokens)

        # Protect the summary and trim `recent`'s oldest messages first
        # instead of handing [summary, *recent] to sliding_window_eviction
        # — that function evicts from the *front* of the list, and the
        # summary sits at the front, so it would be the very first thing
        # dropped. Under a tight budget that silently throws away the
        # summary itself (the whole point of summarizing) and leaves just
        # the last raw message. Evicting from `recent` instead keeps the
        # summarized older context intact for as long as any budget for
        # recent messages remains.
        kept_recent: deque[Message] = deque(recent)
        total = summary_tokens + estimate_total_tokens(recent)
        while total > max_tokens and kept_recent:
            removed = kept_recent.popleft()
            total -= estimate_message_tokens(removed)
        return [summary_message, *kept_recent]

    return policy
