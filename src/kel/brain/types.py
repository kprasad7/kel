from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Route(BaseModel):
    target: str
    confidence: float
    tier: Literal["fast", "slow"]
    reason: str | None = None
