"""Provider-agnostic message, content, and response types.

Every provider adapter translates to/from these types at its boundary, so
nothing above the Model Gateway ever touches a provider-specific SDK shape.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: str
    """Base64-encoded image bytes."""


class ToolUsePart(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentPart = Annotated[
    TextPart | ImagePart | ToolUsePart | ToolResultPart,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Role
    content: list[ContentPart]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[TextPart(text=text)])

    @property
    def text(self) -> str:
        """Concatenation of all text parts, for the common no-multimodal case."""
        return "".join(part.text for part in self.content if isinstance(part, TextPart))


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


StopReason = Literal["end_turn", "max_tokens", "tool_use", "stop_sequence", "error"]


class ModelResponse(BaseModel):
    id: str
    model: str
    content: list[ContentPart]
    stop_reason: StopReason
    usage: Usage
    raw: Any = Field(default=None, repr=False, exclude=True)
    """Untouched provider response, kept for observability/debugging only."""

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> list[ToolUsePart]:
        return [part for part in self.content if isinstance(part, ToolUsePart)]


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallDelta(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    tool_call: ToolUsePart


class MessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"
    response: ModelResponse


StreamEvent = Annotated[
    TextDelta | ToolCallDelta | MessageStop,
    Field(discriminator="type"),
]
