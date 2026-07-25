"""Structured output — the kel equivalent of LangChain's
`.with_structured_output()`. Implemented once, at the function level, on
top of the tool-calling abstraction every provider adapter already
normalizes — no provider-specific "JSON mode" wiring needed.

Mechanism: gives the model exactly one tool (`return_structured_output`,
schema = your pydantic model's JSON schema) and asks it to call that tool
with its answer. If the model doesn't call it, or calls it with data that
fails validation, the error is fed back as a tool result and the model
gets another attempt — bounded by `max_retries`."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from kel.models.base import ChatModel
from kel.models.errors import KelError
from kel.models.types import Message, Role, ToolResultPart, ToolSpec

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "return_structured_output"


class StructuredOutputError(KelError):
    def __init__(self, message: str, *, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def _build_tool(schema: type[BaseModel]) -> ToolSpec:
    return ToolSpec(
        name=_TOOL_NAME,
        description="Return the final answer as structured data matching the required schema.",
        input_schema=schema.model_json_schema(),
    )


def _retry_messages(working: list[Message], response_content, feedback_part) -> list[Message]:
    return [*working, Message(role=Role.ASSISTANT, content=response_content), Message(role=Role.USER, content=[feedback_part])]


def generate_structured(
    model: ChatModel,
    messages: list[Message],
    schema: type[T],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    max_retries: int = 2,
) -> T:
    tool = _build_tool(schema)
    working = list(messages)
    last_error: Exception | None = None

    for _ in range(max_retries + 1):
        response = model.generate(
            working, system=system, tools=[tool], max_tokens=max_tokens, temperature=temperature
        )
        calls = [c for c in response.tool_calls if c.name == _TOOL_NAME]
        if not calls:
            working = [
                *working,
                Message(role=Role.ASSISTANT, content=response.content),
                Message.user(f"You must respond by calling the {_TOOL_NAME} tool, not plain text."),
            ]
            continue
        call = calls[0]
        try:
            return schema.model_validate(call.input)
        except ValidationError as exc:
            last_error = exc
            working = _retry_messages(
                working,
                response.content,
                ToolResultPart(
                    tool_use_id=call.id,
                    content=f"Validation error: {exc}. Please call {_TOOL_NAME} again with corrected data.",
                    is_error=True,
                ),
            )

    raise StructuredOutputError(
        f"failed to get valid structured output ({schema.__name__}) after {max_retries + 1} attempt(s)",
        last_error=last_error,
    )


async def agenerate_structured(
    model: ChatModel,
    messages: list[Message],
    schema: type[T],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    max_retries: int = 2,
) -> T:
    tool = _build_tool(schema)
    working = list(messages)
    last_error: Exception | None = None

    for _ in range(max_retries + 1):
        response = await model.agenerate(
            working, system=system, tools=[tool], max_tokens=max_tokens, temperature=temperature
        )
        calls = [c for c in response.tool_calls if c.name == _TOOL_NAME]
        if not calls:
            working = [*working, Message(role=Role.ASSISTANT, content=response.content), Message.user(f"You must respond by calling the {_TOOL_NAME} tool, not plain text.")]
            continue
        call = calls[0]
        try:
            return schema.model_validate(call.input)
        except ValidationError as exc:
            last_error = exc
            working = _retry_messages(
                working,
                response.content,
                ToolResultPart(
                    tool_use_id=call.id,
                    content=f"Validation error: {exc}. Please call {_TOOL_NAME} again with corrected data.",
                    is_error=True,
                ),
            )

    raise StructuredOutputError(
        f"failed to get valid structured output ({schema.__name__}) after {max_retries + 1} attempt(s)",
        last_error=last_error,
    )
