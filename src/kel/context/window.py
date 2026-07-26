from __future__ import annotations

from kel.context.eviction import EvictionPolicy, sliding_window_eviction
from kel.context.tokens import estimate_message_tokens, estimate_total_tokens
from kel.models.types import Message


class ContextWindow:
    """Tracks a message list against a token budget and applies an
    eviction policy automatically on overflow (DESIGN.md 3.2).

    `tokens_used` is a running total maintained incrementally on every
    `add`/`extend` rather than resummed from scratch each time. This is
    `Memory.working` — the object every `Agent.run()` turn appends to —
    so recomputing over the full history on every append would make an
    n-turn session cost O(n^2) instead of O(n) amortized (the eviction
    path below is the one place a full O(n) recompute is unavoidable,
    since a policy can replace the message list arbitrarily, but that
    only runs on overflow, not on every add).
    """

    def __init__(
        self,
        messages: list[Message] | None = None,
        *,
        max_tokens: int,
        policy: EvictionPolicy = sliding_window_eviction,
    ):
        self.max_tokens = max_tokens
        self.policy = policy
        self.messages: list[Message] = list(messages or [])
        self._evicted_count = 0
        self._tokens_used = estimate_total_tokens(self.messages)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def tokens_remaining(self) -> int:
        return self.max_tokens - self._tokens_used

    @property
    def evicted_count(self) -> int:
        """How many messages have been evicted (dropped or folded into a
        summary) over this window's lifetime — a signal worth tracing."""
        return self._evicted_count

    def add(self, message: Message) -> None:
        self.messages.append(message)
        self._tokens_used += estimate_message_tokens(message)
        self._enforce()

    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
        self._tokens_used += sum(estimate_message_tokens(m) for m in messages)
        self._enforce()

    def _enforce(self) -> None:
        if self._tokens_used <= self.max_tokens:
            return
        before = len(self.messages)
        self.messages = self.policy(self.messages, self.max_tokens)
        self._tokens_used = estimate_total_tokens(self.messages)
        self._evicted_count += max(0, before - len(self.messages))
