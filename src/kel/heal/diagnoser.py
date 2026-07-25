"""LLM-based diagnosis using the user's own configured model (their
kel.models gateway credentials, per DESIGN.md 3.15) — diagnosis cost/
quality is the user's choice, not kel's."""

from __future__ import annotations

from typing import Any

from kel.heal.types import Diagnosis, Strategy
from kel.models.base import ChatModel
from kel.models.types import Message

_VALID_STRATEGIES: frozenset[Strategy] = frozenset(
    {"retry", "fallback_model", "narrow_scope", "escalate_human"}
)


def parse_diagnosis(text: str) -> Diagnosis:
    """Fail-safe parser: anything unparseable defaults to escalate_human
    rather than guessing a retry is safe."""
    strategy: Strategy = "escalate_human"
    reason = text.strip()[:300] or "unparseable diagnosis response"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("STRATEGY:"):
            candidate = stripped.split(":", 1)[1].strip().lower()
            if candidate in _VALID_STRATEGIES:
                strategy = candidate  # type: ignore[assignment]
        elif stripped.upper().startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    return Diagnosis(strategy=strategy, reason=reason)


def make_llm_diagnoser(model: ChatModel):
    def diagnose(error: Exception, context: dict[str, Any]) -> Diagnosis:
        prompt = (
            "A step failed during an agent run. Diagnose the root cause and "
            "choose exactly one recovery strategy.\n\n"
            f"Description: {context.get('description', '')}\n"
            f"Attempt: {context.get('attempt')}\n"
            f"Error: {error}\n\n"
            "Strategies: retry (transient failure, safe to just try again), "
            "fallback_model (this model/provider is the problem), "
            "narrow_scope (the request was too large/ambitious, retry smaller), "
            "escalate_human (none of the above is safe).\n\n"
            "Respond in exactly this format:\nSTRATEGY: <strategy>\nREASON: <one sentence>"
        )
        response = model.generate([Message.user(prompt)])
        return parse_diagnosis(response.text)

    return diagnose
