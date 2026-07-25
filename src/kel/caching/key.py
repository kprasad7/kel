"""Cache key derivation. Deliberately covers only `generate()`'s named
parameters (messages, system, tools, max_tokens, temperature) — provider-
specific `**kwargs` extras aren't included, so two calls differing only in
an extra kwarg would collide on the same key. Known simplification, not a
silent one: don't rely on caching for calls that vary provider-specific
kwargs you care about."""

from __future__ import annotations

import hashlib
import json

from kel.models.types import Message, ToolSpec


def make_cache_key(
    *,
    provider: str,
    model_id: str,
    messages: list[Message],
    system: str | None,
    tools: list[ToolSpec] | None,
    max_tokens: int,
    temperature: float | None,
) -> str:
    payload = {
        "provider": provider,
        "model_id": model_id,
        "messages": [m.model_dump(mode="json") for m in messages],
        "system": system,
        "tools": [t.model_dump(mode="json") for t in tools] if tools else None,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
