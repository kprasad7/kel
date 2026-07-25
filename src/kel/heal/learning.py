"""Closes DESIGN.md 3.15's "learns over time" loop concretely: feed a
Healer's log into a kel.brain.EmbeddingRouter so a future error that looks
like one seen before gets routed to the same strategy without re-diagnosing
from scratch via an LLM call."""

from __future__ import annotations

from collections.abc import Callable

from kel.brain.router import EmbeddingRouter
from kel.heal.types import HealAttempt


def feed_heal_log_into_router(
    log: list[HealAttempt], router: EmbeddingRouter, *, text_fn: Callable[[HealAttempt], str] | None = None
) -> None:
    text_fn = text_fn or (lambda attempt: attempt.error)
    for attempt in log:
        router.add_example(text_fn(attempt), attempt.diagnosis.strategy)
