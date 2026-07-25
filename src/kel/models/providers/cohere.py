"""Cohere adapter (Chat API v2 — `cohere.ClientV2`). Requires the `cohere`
package (`pip install kel[cohere]`).

Caveat, stated plainly: this is written against Cohere's documented v2
Chat API shape (`response.message.content` / `.tool_calls`, streaming
events like `content-delta` / `tool-call-delta` / `message-end`), the same
way the Anthropic/OpenAI adapters are written against their documented
shapes. It has not been exercised against a live Cohere account — if the
installed SDK version's response objects differ, this is the file to
adjust; the request/response mapping is isolated here exactly so that
adjustment doesn't ripple anywhere else."""

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

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "COMPLETE": "end_turn",
    "STOP_SEQUENCE": "stop_sequence",
    "MAX_TOKENS": "max_tokens",
    "TOOL_CALL": "tool_use",
    "ERROR": "error",
}


def _to_cohere_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
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


def _to_cohere_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }
        for t in tools
    ]


def _from_cohere_message(message: Any) -> list[Any]:
    parts: list[Any] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(TextPart(text=block.text))
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(
            ToolUsePart(id=call.id, name=call.function.name, input=json.loads(call.function.arguments or "{}"))
        )
    return parts


class _CohereStreamAccumulator:
    """Shared event-accumulation logic for sync `stream()` and async
    `astream()` — same rationale as OpenAI's accumulator: assemble
    fragmented tool-call events once, use from both code paths."""

    def __init__(self, model_id: str):
        self.text_chunks: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage = Usage()
        self.model_id = model_id

    def process_event(self, event: Any) -> str | None:
        event_type = getattr(event, "type", None)
        delta = getattr(event, "delta", None)
        if delta is None:
            return None

        if event_type == "content-delta":
            text = delta.message.content.text
            self.text_chunks.append(text)
            return text

        if event_type in ("tool-call-start", "tool-call-delta"):
            call = delta.message.tool_calls
            index = getattr(call, "index", 0) or 0
            slot = self.tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if getattr(call, "id", None):
                slot["id"] = call.id
            if getattr(call.function, "name", None):
                slot["name"] = call.function.name
            if getattr(call.function, "arguments", None):
                slot["arguments"] += call.function.arguments
            return None

        if event_type == "message-end":
            self.finish_reason = delta.finish_reason
            billed = delta.usage.billed_units
            self.usage = Usage(
                input_tokens=int(billed.input_tokens or 0), output_tokens=int(billed.output_tokens or 0)
            )
        return None

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
            id="",
            model=self.model_id,
            content=content,
            stop_reason=_FINISH_REASON_MAP.get(self.finish_reason or "", "error"),
            usage=self.usage,
            raw=None,
        )
        return tool_use_parts, response


def _translate_error(exc: Exception, *, provider: str) -> ProviderError:
    import cohere

    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 401 or isinstance(exc, getattr(cohere, "UnauthorizedError", ())):
        return AuthenticationError(str(exc), provider=provider)
    if status == 429 or isinstance(exc, getattr(cohere, "TooManyRequestsError", ())):
        return RateLimitError(str(exc), provider=provider)
    return ProviderError(str(exc), provider=provider, retryable=True)


class CohereChatModel(ChatModel):
    provider = "cohere"

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
                import cohere
            except ImportError as exc:
                raise ImportError(
                    "The cohere package is required to use CohereChatModel. "
                    "Install it with `pip install kel[cohere]`."
                ) from exc
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def _ensure_async_client(self) -> Any:
        if self._async_client is None:
            try:
                import cohere
            except ImportError as exc:
                raise ImportError(
                    "The cohere package is required to use CohereChatModel. "
                    "Install it with `pip install kel[cohere]`."
                ) from exc
            self._async_client = cohere.AsyncClientV2(api_key=self._api_key)
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
            "messages": _to_cohere_messages(messages, system),
        }
        if tools:
            params["tools"] = _to_cohere_tools(tools)
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
            resp = client.chat(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        billed = resp.usage.billed_units
        return ModelResponse(
            id=resp.id,
            model=self.model_id,
            content=_from_cohere_message(resp.message),
            stop_reason=_FINISH_REASON_MAP.get(resp.finish_reason, "error"),
            usage=Usage(
                input_tokens=int(billed.input_tokens or 0), output_tokens=int(billed.output_tokens or 0)
            ),
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
        acc = _CohereStreamAccumulator(self.model_id)
        try:
            for event in client.chat_stream(**params):
                text = acc.process_event(event)
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
            resp = await client.chat(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        billed = resp.usage.billed_units
        return ModelResponse(
            id=resp.id,
            model=self.model_id,
            content=_from_cohere_message(resp.message),
            stop_reason=_FINISH_REASON_MAP.get(resp.finish_reason, "error"),
            usage=Usage(
                input_tokens=int(billed.input_tokens or 0), output_tokens=int(billed.output_tokens or 0)
            ),
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
        acc = _CohereStreamAccumulator(self.model_id)
        try:
            async for event in client.chat_stream(**params):
                text = acc.process_event(event)
                if text:
                    yield TextDelta(text=text)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        tool_use_parts, response = acc.finalize()
        for tool_use in tool_use_parts:
            yield ToolCallDelta(tool_call=tool_use)
        yield MessageStop(response=response)


class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embeddings (v2 `embed` endpoint). `input_type` matters for
    embedding quality — Cohere's v3+ embedding models are trained with
    different embeddings for "this is a document to index" vs "this is a
    search query", so mismatching them silently degrades retrieval
    relevance rather than erroring."""

    provider = "cohere"

    def __init__(
        self,
        model_id: str = "embed-v4.0",
        *,
        api_key: str | None = None,
        client: Any = None,
        input_type: str = "search_document",
    ):
        self.model_id = model_id
        self._api_key = api_key
        self._client = client
        self.input_type = input_type

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import cohere
            except ImportError as exc:
                raise ImportError(
                    "The cohere package is required to use CohereEmbeddingModel. "
                    "Install it with `pip install kel[cohere]`."
                ) from exc
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        try:
            resp = client.embed(
                model=self.model_id, texts=texts, input_type=self.input_type, embedding_types=["float"]
            )
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc
        return resp.embeddings.float_
