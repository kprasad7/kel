"""OpenAI adapter (Chat Completions API). Requires the `openai` package
(`pip install kel[openai]`).

OpenAI's wire format splits what kel represents as content parts across
different fields (assistant tool calls live in `tool_calls`, not `content`;
tool results are a separate `role: "tool"` message, not an inline part) —
this adapter is where that translation happens so nothing above it needs
to know OpenAI's shape.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.models.base import ChatModel, EmbeddingModel
from kel.models.errors import AuthenticationError, ProviderError, RateLimitError
from kel.models.types import (
    ImagePart,
    Message,
    MessageStop,
    ModelResponse,
    Role,
    StreamEvent,
    StopReason,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolResultPart,
    ToolSpec,
    ToolUsePart,
    Usage,
)

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "error",
}


def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for m in messages:
        if m.role == Role.ASSISTANT:
            text = "".join(p.text for p in m.content if isinstance(p, TextPart))
            tool_uses = [p for p in m.content if isinstance(p, ToolUsePart)]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_uses:
                msg["tool_calls"] = [
                    {
                        "id": t.id,
                        "type": "function",
                        "function": {"name": t.name, "arguments": json.dumps(t.input)},
                    }
                    for t in tool_uses
                ]
            out.append(msg)
            continue

        # user role: text/image parts become one user message; tool results
        # each become their own role="tool" message, per OpenAI's wire format.
        content_items: list[dict[str, Any]] = []
        for p in m.content:
            if isinstance(p, TextPart):
                content_items.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                content_items.append(
                    {"type": "image_url", "image_url": {"url": f"data:{p.media_type};base64,{p.data}"}}
                )
            elif isinstance(p, ToolResultPart):
                out.append({"role": "tool", "tool_call_id": p.tool_use_id, "content": p.content})
        if content_items:
            out.append({"role": "user", "content": content_items})

    return out


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }
        for t in tools
    ]


def _from_openai_message(message: Any) -> list[Any]:
    parts: list[Any] = []
    if message.content:
        parts.append(TextPart(text=message.content))
    for call in message.tool_calls or []:
        parts.append(
            ToolUsePart(id=call.id, name=call.function.name, input=json.loads(call.function.arguments or "{}"))
        )
    return parts


class _StreamAccumulator:
    """Shared chunk-accumulation logic for sync `stream()` and async
    `astream()` — OpenAI's streaming chunks arrive as fragments (tool-call
    arguments split across many chunks, indexed by position) regardless of
    sync/async, so this is the one place that assembles them."""

    def __init__(self, model_id: str):
        self.text_chunks: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage = Usage()
        self.resp_id = ""
        self.model = model_id

    def process_chunk(self, chunk: Any) -> str | None:
        self.resp_id = chunk.id or self.resp_id
        self.model = chunk.model or self.model
        if chunk.usage:
            self.usage = Usage(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
        if not chunk.choices:
            return None
        choice = chunk.choices[0]
        self.finish_reason = choice.finish_reason or self.finish_reason
        delta = choice.delta
        text_out = delta.content if delta.content else None
        if text_out:
            self.text_chunks.append(text_out)
        for call_delta in delta.tool_calls or []:
            slot = self.tool_calls.setdefault(call_delta.index, {"id": "", "name": "", "arguments": ""})
            if call_delta.id:
                slot["id"] = call_delta.id
            if call_delta.function and call_delta.function.name:
                slot["name"] = call_delta.function.name
            if call_delta.function and call_delta.function.arguments:
                slot["arguments"] += call_delta.function.arguments
        return text_out

    def finalize(self) -> tuple[list[ToolUsePart], ModelResponse]:
        content: list[Any] = []
        if self.text_chunks:
            content.append(TextPart(text="".join(self.text_chunks)))
        tool_use_parts: list[ToolUsePart] = []
        for slot in self.tool_calls.values():
            tool_use = ToolUsePart(id=slot["id"], name=slot["name"], input=json.loads(slot["arguments"] or "{}"))
            content.append(tool_use)
            tool_use_parts.append(tool_use)
        response = ModelResponse(
            id=self.resp_id,
            model=self.model,
            content=content,
            stop_reason=_FINISH_REASON_MAP.get(self.finish_reason or "", "error"),
            usage=self.usage,
            raw=None,
        )
        return tool_use_parts, response


def _translate_error(exc: Exception, *, provider: str) -> ProviderError:
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider)
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(str(exc), provider=provider)
    if isinstance(exc, openai.APIError):
        return ProviderError(str(exc), provider=provider, retryable=True)
    return ProviderError(str(exc), provider=provider, retryable=False)


class OpenAIChatModel(ChatModel):
    provider = "openai"

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
                import openai
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required to use OpenAIChatModel. "
                    "Install it with `pip install kel[openai]`."
                ) from exc
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _ensure_async_client(self) -> Any:
        if self._async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required to use OpenAIChatModel. "
                    "Install it with `pip install kel[openai]`."
                ) from exc
            self._async_client = openai.AsyncOpenAI(api_key=self._api_key)
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
            "messages": _to_openai_messages(messages, system),
        }
        if tools:
            params["tools"] = _to_openai_tools(tools)
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
            resp = client.chat.completions.create(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        choice = resp.choices[0]
        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_openai_message(choice.message),
            stop_reason=_FINISH_REASON_MAP.get(choice.finish_reason, "error"),
            usage=Usage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens),
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
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        client = self._ensure_client()
        acc = _StreamAccumulator(self.model_id)
        try:
            for chunk in client.chat.completions.create(**params):
                text = acc.process_chunk(chunk)
                if text:
                    yield TextDelta(text=text)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        tool_use_parts, response = acc.finalize()
        for tool_use in tool_use_parts:
            yield ToolCallDelta(tool_call=tool_use)
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
            resp = await client.chat.completions.create(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        choice = resp.choices[0]
        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_openai_message(choice.message),
            stop_reason=_FINISH_REASON_MAP.get(choice.finish_reason, "error"),
            usage=Usage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens),
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
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        client = self._ensure_async_client()
        acc = _StreamAccumulator(self.model_id)
        try:
            stream = await client.chat.completions.create(**params)
            async for chunk in stream:
                text = acc.process_chunk(chunk)
                if text:
                    yield TextDelta(text=text)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        tool_use_parts, response = acc.finalize()
        for tool_use in tool_use_parts:
            yield ToolCallDelta(tool_call=tool_use)
        yield MessageStop(response=response)


class OpenAIEmbeddingModel(EmbeddingModel):
    provider = "openai"

    def __init__(self, model_id: str = "text-embedding-3-small", *, api_key: str | None = None, client: Any = None):
        self.model_id = model_id
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required to use OpenAIEmbeddingModel. "
                    "Install it with `pip install kel[openai]`."
                ) from exc
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        try:
            resp = client.embeddings.create(model=self.model_id, input=texts)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc
        return [item.embedding for item in resp.data]
