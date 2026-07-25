"""Anthropic adapter. Requires the `anthropic` package (`pip install kel[anthropic]`)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.models.base import ChatModel
from kel.models.errors import AuthenticationError, ProviderError, RateLimitError
from kel.models.types import (
    ImagePart,
    Message,
    MessageStop,
    ModelResponse,
    StopReason,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolResultPart,
    ToolSpec,
    ToolUsePart,
    Usage,
)

_VALID_STOP_REASONS = {"end_turn", "max_tokens", "tool_use", "stop_sequence"}


def _to_anthropic_content(parts: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
                }
            )
        elif isinstance(part, ToolUsePart):
            blocks.append({"type": "tool_use", "id": part.id, "name": part.name, "input": part.input})
        elif isinstance(part, ToolResultPart):
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": part.tool_use_id,
                    "content": part.content,
                    "is_error": part.is_error,
                }
            )
    return blocks


def _from_anthropic_content(blocks: list[Any]) -> list[Any]:
    parts: list[Any] = []
    for block in blocks:
        if block.type == "text":
            parts.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            parts.append(ToolUsePart(id=block.id, name=block.name, input=block.input))
    return parts


def _map_stop_reason(reason: str | None) -> StopReason:
    if reason in _VALID_STOP_REASONS:
        return reason  # type: ignore[return-value]
    return "error"


def _translate_error(exc: Exception, *, provider: str) -> ProviderError:
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider)
    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc), provider=provider)
    if isinstance(exc, anthropic.APIError):
        return ProviderError(str(exc), provider=provider, retryable=True)
    return ProviderError(str(exc), provider=provider, retryable=False)


class AnthropicChatModel(ChatModel):
    provider = "anthropic"

    def __init__(
        self, model_id: str, *, api_key: str | None = None, client: Any = None, async_client: Any = None
    ):
        self.model_id = model_id
        self._api_key = api_key
        self._client = client
        self._async_client = async_client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "The anthropic package is required to use AnthropicChatModel. "
                    "Install it with `pip install kel[anthropic]`."
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _ensure_async_client(self) -> Any:
        if self._async_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "The anthropic package is required to use AnthropicChatModel. "
                    "Install it with `pip install kel[anthropic]`."
                ) from exc
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    def _build_params(
        self,
        messages: list[Message],
        *,
        system: str | None,
        tools: list[ToolSpec] | None,
        max_tokens: int,
        temperature: float | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": m.role.value, "content": _to_anthropic_content(m.content)} for m in messages
            ],
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        if temperature is not None:
            params["temperature"] = temperature
        params.update(extra)
        return params

    def generate(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        params = self._build_params(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs
        )
        client = self._ensure_client()
        try:
            resp = client.messages.create(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_anthropic_content(resp.content),
            stop_reason=_map_stop_reason(resp.stop_reason),
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            raw=resp,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        params = self._build_params(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs
        )
        client = self._ensure_client()
        try:
            with client.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                        yield TextDelta(text=event.delta.text)
                final = stream.get_final_message()
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content = _from_anthropic_content(final.content)
        for part in content:
            if isinstance(part, ToolUsePart):
                yield ToolCallDelta(tool_call=part)

        response = ModelResponse(
            id=final.id,
            model=final.model,
            content=content,
            stop_reason=_map_stop_reason(final.stop_reason),
            usage=Usage(input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens),
            raw=final,
        )
        yield MessageStop(response=response)

    async def agenerate(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        params = self._build_params(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs
        )
        client = self._ensure_async_client()
        try:
            resp = await client.messages.create(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_anthropic_content(resp.content),
            stop_reason=_map_stop_reason(resp.stop_reason),
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            raw=resp,
        )

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        params = self._build_params(
            messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs
        )
        client = self._ensure_async_client()
        try:
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                        yield TextDelta(text=event.delta.text)
                final = await stream.get_final_message()
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content = _from_anthropic_content(final.content)
        for part in content:
            if isinstance(part, ToolUsePart):
                yield ToolCallDelta(tool_call=part)

        response = ModelResponse(
            id=final.id,
            model=final.model,
            content=content,
            stop_reason=_map_stop_reason(final.stop_reason),
            usage=Usage(input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens),
            raw=final,
        )
        yield MessageStop(response=response)
