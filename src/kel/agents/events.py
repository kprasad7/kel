"""Agent-loop-level stream events. A raw `ChatModel.stream()` only covers
one model call; `Agent.run_stream()`/`arun_stream()` span the whole
multi-turn tool loop, so a `ToolResultEvent` is added to announce a tool
finished executing between model calls — the piece a UI needs to show
"calling search..." progress that plain model-level streaming can't
express on its own."""

from __future__ import annotations

from pydantic import BaseModel

from kel.models.types import ToolResultPart


class ToolResultEvent(BaseModel):
    tool_use_id: str
    name: str
    result: ToolResultPart
