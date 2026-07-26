"""Shared StreamEvent -> JSON serialization for kel.sdk's streaming
transports (WebSocket, FastAPI SSE) — one translation, not one per
transport."""

from __future__ import annotations

from typing import Any

from kel.agents.events import ToolResultEvent
from kel.models.types import MessageStop, TextDelta, ToolCallDelta


def event_to_json(event: Any) -> dict[str, Any]:
    if isinstance(event, TextDelta):
        return {"type": "text_delta", "text": event.text}
    if isinstance(event, ToolCallDelta):
        return {"type": "tool_call_delta", "name": event.tool_call.name, "input": event.tool_call.input}
    if isinstance(event, ToolResultEvent):
        return {
            "type": "tool_result",
            "name": event.name,
            "result": event.result.content,
            "is_error": event.result.is_error,
        }
    if isinstance(event, MessageStop):
        return {"type": "message_stop", "text": event.response.text, "stop_reason": event.response.stop_reason}
    return {"type": "unknown"}
