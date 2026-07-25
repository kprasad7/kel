"""Gemini adapter (`google-genai` SDK). Requires `pip install kel[gemini]`.

Same documented-shape caveat as Cohere: written against `google-genai`'s
documented client API, tested here against an injected fake client, not
exercised against a live Gemini account.

**Credentials are optional by design.** If `api_key` isn't given, this
never forces one — the underlying client is constructed with no explicit
key, so `google-genai` falls back to the `GEMINI_API_KEY`/`GOOGLE_API_KEY`
env vars, or — when running on GCP (GKE with Workload Identity, GCE, Cloud
Run) — Application Default Credentials with no key at all. That ambient-
credential path is exactly what running in EKS/GKE/EC2 with an attached
role needs; forcing an explicit key argument would break it.
"""

from __future__ import annotations

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
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
}


def _to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = "model" if m.role == Role.ASSISTANT else "user"
        parts: list[dict[str, Any]] = []
        for p in m.content:
            if isinstance(p, TextPart):
                parts.append({"text": p.text})
            elif isinstance(p, ImagePart):
                parts.append({"inline_data": {"mime_type": p.media_type, "data": p.data}})
            elif isinstance(p, ToolUsePart):
                parts.append({"function_call": {"name": p.name, "args": p.input}})
            elif isinstance(p, ToolResultPart):
                parts.append(
                    {"function_response": {"name": p.tool_use_id, "response": {"result": p.content}}}
                )
        contents.append({"role": role, "parts": parts})
    return contents


def _to_gemini_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "function_declarations": [
                {"name": t.name, "description": t.description, "parameters": t.input_schema} for t in tools
            ]
        }
    ]


def _from_gemini_parts(parts: list[Any]) -> list[Any]:
    out: list[Any] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            out.append(TextPart(text=text))
        func_call = getattr(part, "function_call", None)
        if func_call is not None:
            out.append(ToolUsePart(id=func_call.name, name=func_call.name, input=dict(func_call.args or {})))
    return out


def _map_finish_reason(reason: Any, has_tool_call: bool) -> StopReason:
    if has_tool_call:
        return "tool_use"
    return _FINISH_REASON_MAP.get(str(reason), "error")


def _translate_error(exc: Exception, *, provider: str) -> ProviderError:
    message = str(exc)
    lowered = message.lower()
    if "permission" in lowered or "unauthenticated" in lowered or "api key not valid" in lowered:
        return AuthenticationError(message, provider=provider)
    if "429" in message or "rate limit" in lowered or "resource_exhausted" in lowered:
        return RateLimitError(message, provider=provider)
    return ProviderError(message, provider=provider, retryable=True)


class GeminiChatModel(ChatModel):
    provider = "gemini"

    def __init__(self, model_id: str, *, api_key: str | None = None, client: Any = None):
        self.model_id = model_id
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "The google-genai package is required to use GeminiChatModel. "
                    "Install it with `pip install kel[gemini]`."
                ) from exc
            # api_key=None lets the client fall back to env vars / ADC — never force a key here.
            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._client

    def _build_config(
        self, *, system: str | None, tools: list[ToolSpec] | None, max_tokens: int, temperature: float | None
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = _to_gemini_tools(tools)
        if temperature is not None:
            config["temperature"] = temperature
        return config

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
        client = self._ensure_client()
        config = self._build_config(system=system, tools=tools, max_tokens=max_tokens, temperature=temperature)
        config.update(kwargs)
        try:
            resp = client.models.generate_content(
                model=self.model_id, contents=_to_gemini_contents(messages), config=config
            )
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content = _from_gemini_parts(resp.candidates[0].content.parts)
        has_tool_call = any(isinstance(p, ToolUsePart) for p in content)
        return ModelResponse(
            id=getattr(resp, "response_id", "") or "",
            model=self.model_id,
            content=content,
            stop_reason=_map_finish_reason(resp.candidates[0].finish_reason, has_tool_call),
            usage=Usage(
                input_tokens=resp.usage_metadata.prompt_token_count or 0,
                output_tokens=resp.usage_metadata.candidates_token_count or 0,
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
        client = self._ensure_client()
        config = self._build_config(system=system, tools=tools, max_tokens=max_tokens, temperature=temperature)
        config.update(kwargs)

        text_chunks: list[str] = []
        tool_parts: list[ToolUsePart] = []
        usage = Usage()
        finish_reason: Any = None
        try:
            for chunk in client.models.generate_content_stream(
                model=self.model_id, contents=_to_gemini_contents(messages), config=config
            ):
                candidate = chunk.candidates[0] if chunk.candidates else None
                if candidate is None:
                    continue
                finish_reason = candidate.finish_reason or finish_reason
                for part in _from_gemini_parts(candidate.content.parts):
                    if isinstance(part, TextPart):
                        text_chunks.append(part.text)
                        yield TextDelta(text=part.text)
                    elif isinstance(part, ToolUsePart):
                        tool_parts.append(part)
                if getattr(chunk, "usage_metadata", None):
                    usage = Usage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    )
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content: list[Any] = []
        if text_chunks:
            content.append(TextPart(text="".join(text_chunks)))
        for tool_part in tool_parts:
            content.append(tool_part)
            yield ToolCallDelta(tool_call=tool_part)

        response = ModelResponse(
            id="",
            model=self.model_id,
            content=content,
            stop_reason=_map_finish_reason(finish_reason, bool(tool_parts)),
            usage=usage,
            raw=None,
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
        client = self._ensure_client()
        config = self._build_config(system=system, tools=tools, max_tokens=max_tokens, temperature=temperature)
        config.update(kwargs)
        try:
            resp = await client.aio.models.generate_content(
                model=self.model_id, contents=_to_gemini_contents(messages), config=config
            )
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content = _from_gemini_parts(resp.candidates[0].content.parts)
        has_tool_call = any(isinstance(p, ToolUsePart) for p in content)
        return ModelResponse(
            id=getattr(resp, "response_id", "") or "",
            model=self.model_id,
            content=content,
            stop_reason=_map_finish_reason(resp.candidates[0].finish_reason, has_tool_call),
            usage=Usage(
                input_tokens=resp.usage_metadata.prompt_token_count or 0,
                output_tokens=resp.usage_metadata.candidates_token_count or 0,
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
        client = self._ensure_client()
        config = self._build_config(system=system, tools=tools, max_tokens=max_tokens, temperature=temperature)
        config.update(kwargs)

        text_chunks: list[str] = []
        tool_parts: list[ToolUsePart] = []
        usage = Usage()
        finish_reason: Any = None
        try:
            stream = await client.aio.models.generate_content_stream(
                model=self.model_id, contents=_to_gemini_contents(messages), config=config
            )
            async for chunk in stream:
                candidate = chunk.candidates[0] if chunk.candidates else None
                if candidate is None:
                    continue
                finish_reason = candidate.finish_reason or finish_reason
                for part in _from_gemini_parts(candidate.content.parts):
                    if isinstance(part, TextPart):
                        text_chunks.append(part.text)
                        yield TextDelta(text=part.text)
                    elif isinstance(part, ToolUsePart):
                        tool_parts.append(part)
                if getattr(chunk, "usage_metadata", None):
                    usage = Usage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    )
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc

        content: list[Any] = []
        if text_chunks:
            content.append(TextPart(text="".join(text_chunks)))
        for tool_part in tool_parts:
            content.append(tool_part)
            yield ToolCallDelta(tool_call=tool_part)

        response = ModelResponse(
            id="",
            model=self.model_id,
            content=content,
            stop_reason=_map_finish_reason(finish_reason, bool(tool_parts)),
            usage=usage,
            raw=None,
        )
        yield MessageStop(response=response)


class GeminiEmbeddingModel(EmbeddingModel):
    provider = "gemini"

    def __init__(self, model_id: str = "text-embedding-004", *, api_key: str | None = None, client: Any = None):
        self.model_id = model_id
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "The google-genai package is required to use GeminiEmbeddingModel. "
                    "Install it with `pip install kel[gemini]`."
                ) from exc
            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        try:
            resp = client.models.embed_content(model=self.model_id, contents=texts)
        except Exception as exc:
            raise _translate_error(exc, provider=self.provider) from exc
        return [e.values for e in resp.embeddings]
