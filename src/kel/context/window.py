from __future__ import annotations

from kel.context.eviction import EvictionPolicy, sliding_window_eviction
from kel.context.tokens import estimate_total_tokens
from kel.models.types import Message


class ContextWindow:
    """Tracks a message list against a token budget and applies an
    eviction policy automatically on overflow (DESIGN.md 3.2)."""

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

    @property
    def tokens_used(self) -> int:
        return estimate_total_tokens(self.messages)

    @property
    def tokens_remaining(self) -> int:
        return self.max_tokens - self.tokens_used

    @property
    def evicted_count(self) -> int:
        """How many messages have been evicted (dropped or folded into a
        summary) over this window's lifetime — a signal worth tracing."""
        return self._evicted_count

    def add(self, message: Message) -> None:
        self.messages.append(message)
        self._enforce()

    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
        self._enforce()

    def _enforce(self) -> None:
        if self.tokens_used <= self.max_tokens:
            return
        before = len(self.messages)
        self.messages = self.policy(self.messages, self.max_tokens)
        self._evicted_count += max(0, before - len(self.messages))
