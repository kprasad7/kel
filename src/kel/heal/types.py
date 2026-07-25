from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Strategy = Literal["retry", "fallback_model", "narrow_scope", "escalate_human"]


class Diagnosis(BaseModel):
    strategy: Strategy
    reason: str
    corrected_input: dict[str, Any] | None = None


class HealAttempt(BaseModel):
    attempt: int
    error: str
    diagnosis: Diagnosis
    outcome: Literal["retried", "escalated"]
