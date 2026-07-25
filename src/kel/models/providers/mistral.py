"""Mistral adapter (`mistralai` SDK). Requires `pip install kel[mistral]`.

Same documented-shape caveat as Cohere/Gemini: written against the SDK's
documented (OpenAI-similar) chat API, tested against an injected fake
client, not exercised against a live account.

Credentials optional by design: kel reads `MISTRAL_API_KEY` from the
environment itself when `api_key` isn't given, rather than assuming the
SDK does — so this works the same whether the key comes from an explicit
argument, an env var set by a Kubernetes secret, or any other ambient
mechanism your deployment already uses.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.models.base import ChatModel
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
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _to_mistral_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
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
                    {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": json.dumps(t.input)}}
                    for t in tool_uses
                ]
            out.append(msg)
            continue
        content_items: list[dict[str, Any]] = []
        for p in m.content:
            if isinstance(p, TextPart):
                content_items.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                content_items.append({"type": "image_url", "image_url": f"data:{p.media_type};base64,{p.data}"})
            elif isinstance(p, ToolResultPart):
                out.append({"role": "tool", "tool_call_id": p.tool_use_id, "content": p.content})
        if content_items:
            out.append({"role": "user", "content": content_items})
    return out


def _to_mistral_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


def _from_mistral_message(message: Any) -> list[Any]:
    parts: list[Any] = []
    if message.content:
        parts.append(TextPart(text=message.content))
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(ToolUsePart(id=call.id, name=call.function.name, input=json.loads(call.function.arguments or "{}")))
    return parts


def _translate_error(exc: Exception, *, provider: str) -> ProviderError:
    status = getattr(exc, "status_code", None)
    if status == 401:
        return AuthenticationError(str(exc), provider=provider)
    if status == 429:
        return RateLimitError(str(exc), provider=provider)
    return ProviderError(str(exc), provider=provider, retryable=True)


class MistralChatModel(ChatModel):
    provider = "mistral"

    def __init__(self, model_id: str, *, api_key: str | None = None, client: Any = None, async_client: Any = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._client = client
        self._async_client = async_client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from mistralai import Mistral
            except ImportError as exc:
                raise ImportError(
                    "The mistralai package is required to use MistralChatModel. "
                    "Install it with `pip install kel[mistral]`."
                ) from exc
            self._client = Mistral(api_key=self._api_key)
        return self._client

    def _ensure_async_client(self) -> Any:
        # mistralai's client exposes async methods on the same object (chat.complete_async), not a separate client
        return self._ensure_client()

    def _build_params(
        self, messages, *, system, tools, max_tokens, temperature, extra
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": _to_mistral_messages(messages, system),
        }
        if tools:
            params["tools"] = _to_mistral_tools(tools)
        if temperature is not None:
            params["temperature"] = temperature
        params.update(extra)
        return params

    def generate(self, messages, *, system=None, tools=None, max_tokens=1024, temperature=None, **kwargs) -> ModelResponse:
        client = self._ensure_client()
        params = self._build_params(messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs)
        try:
            resp = client.chat.complete(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc
        choice = resp.choices[0]
        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_mistral_message(choice.message),
            stop_reason=_FINISH_REASON_MAP.get(choice.finish_reason, "error"),
            usage=Usage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens),
            raw=resp,
        )

    def stream(self, messages, *, system=None, tools=None, max_tokens=1024, temperature=None, **kwargs) -> Iterator[StreamEvent]:
        client = self._ensure_client()
        params = self._build_params(messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs)
        text_chunks: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage = Usage()
        resp_id = ""
        try:
            for event in client.chat.stream(**params):
                chunk = event.data
                resp_id = chunk.id or resp_id
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta.content:
                    text_chunks.append(delta.content)
                    yield TextDelta(text=delta.content)
                for call_delta in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(call_delta, "index", 0) or 0
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if call_delta.id:
                        slot["id"] = call_delta.id
                    if call_delta.function and call_delta.function.name:
                        slot["name"] = call_delta.function.name
                    if call_delta.function and call_delta.function.arguments:
                        slot["arguments"] += call_delta.function.arguments
                if getattr(chunk, "usage", None):
                    usage = Usage(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content: list[Any] = []
        if text_chunks:
            content.append(TextPart(text="".join(text_chunks)))
        for slot in tool_calls.values():
            tool_use = ToolUsePart(id=slot["id"], name=slot["name"], input=json.loads(slot["arguments"] or "{}"))
            content.append(tool_use)
            yield ToolCallDelta(tool_call=tool_use)
        yield MessageStop(response=ModelResponse(
            id=resp_id, model=self.model_id, content=content,
            stop_reason=_FINISH_REASON_MAP.get(finish_reason or "", "error"), usage=usage, raw=None,
        ))

    async def agenerate(self, messages, *, system=None, tools=None, max_tokens=1024, temperature=None, **kwargs) -> ModelResponse:
        client = self._ensure_async_client()
        params = self._build_params(messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs)
        try:
            resp = await client.chat.complete_async(**params)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc
        choice = resp.choices[0]
        return ModelResponse(
            id=resp.id,
            model=resp.model,
            content=_from_mistral_message(choice.message),
            stop_reason=_FINISH_REASON_MAP.get(choice.finish_reason, "error"),
            usage=Usage(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens),
            raw=resp,
        )

    async def astream(self, messages, *, system=None, tools=None, max_tokens=1024, temperature=None, **kwargs) -> AsyncIterator[StreamEvent]:
        client = self._ensure_async_client()
        params = self._build_params(messages, system=system, tools=tools, max_tokens=max_tokens, temperature=temperature, extra=kwargs)
        text_chunks: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage = Usage()
        resp_id = ""
        try:
            stream = await client.chat.stream_async(**params)
            async for event in stream:
                chunk = event.data
                resp_id = chunk.id or resp_id
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta.content:
                    text_chunks.append(delta.content)
                    yield TextDelta(text=delta.content)
                for call_delta in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(call_delta, "index", 0) or 0
                    slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if call_delta.id:
                        slot["id"] = call_delta.id
                    if call_delta.function and call_delta.function.name:
                        slot["name"] = call_delta.function.name
                    if call_delta.function and call_delta.function.arguments:
                        slot["arguments"] += call_delta.function.arguments
                if getattr(chunk, "usage", None):
                    usage = Usage(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content: list[Any] = []
        if text_chunks:
            content.append(TextPart(text="".join(text_chunks)))
        for slot in tool_calls.values():
            tool_use = ToolUsePart(id=slot["id"], name=slot["name"], input=json.loads(slot["arguments"] or "{}"))
            content.append(tool_use)
            yield ToolCallDelta(tool_call=tool_use)
        yield MessageStop(response=ModelResponse(
            id=resp_id, model=self.model_id, content=content,
            stop_reason=_FINISH_REASON_MAP.get(finish_reason or "", "error"), usage=usage, raw=None,
        ))
