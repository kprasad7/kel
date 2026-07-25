"""Bounded, diagnosis-driven retry harness (DESIGN.md 3.15).

The idempotency guardrail is enforced centrally here, not left for callers
to remember on every call site: a diagnosis that wants to retry, swap
model, or narrow scope is only ever honored when the caller explicitly
passed `idempotent=True`. Otherwise it's downgraded to `escalate_human`
regardless of what the diagnosis said — DESIGN.md §7 calls this
"non-negotiable," and it's implemented that way, not as an opt-in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.context.errors import LoopBudgetExceededError
from kel.context.loop import Loop
from kel.heal.errors import HealExhaustedError
from kel.heal.types import Diagnosis, HealAttempt

Diagnoser = Callable[[Exception, dict[str, Any]], Diagnosis]
RetryStrategies = frozenset({"retry", "fallback_model", "narrow_scope"})


class Healer:
    def __init__(self, *, diagnose: Diagnoser, max_attempts: int = 3):
        self.diagnose = diagnose
        self.max_attempts = max_attempts
        self.log: list[HealAttempt] = []

    def run(self, fn: Callable[[Diagnosis | None], Any], *, idempotent: bool, description: str = "") -> Any:
        """`fn` receives `None` on the first call, then the most recent
        `Diagnosis` on every retry — so it can act on `fallback_model` /
        `narrow_scope` / `corrected_input` by adjusting its own closure."""
        loop = Loop(max_iterations=self.max_attempts)
        attempt_num = 0
        diagnosis_for_retry: Diagnosis | None = None

        while True:
            try:
                loop.step()
            except LoopBudgetExceededError as exc:
                raise HealExhaustedError(
                    f"heal attempts exhausted after {attempt_num} tries", attempts=list(self.log)
                ) from exc

            attempt_num += 1
            try:
                return fn(diagnosis_for_retry)
            except Exception as exc:
                diagnosis = self.diagnose(exc, {"description": description, "attempt": attempt_num})

                if diagnosis.strategy in RetryStrategies and not idempotent:
                    diagnosis = Diagnosis(
                        strategy="escalate_human",
                        reason=f"refused to auto-retry a non-idempotent action (diagnosis was: {diagnosis.reason})",
                    )

                will_retry = diagnosis.strategy in RetryStrategies
                self.log.append(
                    HealAttempt(
                        attempt=attempt_num,
                        error=str(exc),
                        diagnosis=diagnosis,
                        outcome="retried" if will_retry else "escalated",
                    )
                )

                if not will_retry:
                    raise HealExhaustedError(
                        f"heal escalated after attempt {attempt_num}: {diagnosis.reason}", attempts=list(self.log)
                    ) from exc

                diagnosis_for_retry = diagnosis
