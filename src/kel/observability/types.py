from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Span(BaseModel):
    id: str
    parent_id: str | None = None
    name: str
    start_time: float
    """Unix epoch seconds."""
    duration_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
