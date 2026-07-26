"""Shared best-effort HTTP-status-based error translation for `kel.media`
providers. Unlike `kel.models`' chat-model adapters — which pattern-match
each vendor SDK's own, confidently-known exception classes (e.g.
`anthropic.RateLimitError`) — `kel.media`'s providers don't have that same
confidence in their SDKs' exact exception hierarchy, so every provider
here converges on one generic status-code heuristic instead of each
duplicating (and possibly getting wrong) its own vendor-specific guess."""

from __future__ import annotations

from kel.models.errors import AuthenticationError, ProviderError, RateLimitError


def status_code_of(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    for attr in ("status_code", "status"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def translate_error(exc: Exception, *, provider: str) -> ProviderError:
    status = status_code_of(exc)
    if status in (401, 403):
        return AuthenticationError(str(exc), provider=provider)
    if status == 429:
        return RateLimitError(str(exc), provider=provider)
    return ProviderError(str(exc), provider=provider, retryable=status in (500, 502, 503, 504) if status else False)
