"""Generic adapter for fal.ai model endpoints (`pip install kel[fal]`).

Image generation, video generation, text-to-speech, and lipsync are all
the same request shape on fal's platform — submit `arguments` to a named
model endpoint (e.g. `fal-ai/flux/schnell`,
`fal-ai/kling-video/v1.6/standard/text-to-video`,
`fal-ai/elevenlabs/tts/turbo-v2.5`, `fal-ai/sync-lipsync`), get a result
back — so one generic class covers every media type and every "scale"
(image resolution, video duration/fps, voice, etc.) instead of kel
hardcoding one wrapper class and one fixed argument schema per model.
Those per-endpoint arguments are fal/model-specific — pass whatever the
specific endpoint's own docs specify as `**arguments` to `generate()`.

Same dependency-injection shape as every other adapter in kel: pass
`client=` (anything exposing `.run(endpoint, arguments=...)` /
`.run_async(...)`) to test against a fake instead of the real network.
"""

from __future__ import annotations

from typing import Any, Protocol

from kel.media.types import MediaResult
from kel.observability import get_tracer


class _FalClient(Protocol):
    def run(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class _AsyncFalClient(Protocol):
    async def run(self, endpoint: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _import_fal_client() -> Any:
    try:
        import fal_client
    except ImportError as exc:
        raise ImportError(
            "The fal-client package is required for fal.ai media generation. "
            "Install it with `pip install kel[fal]`."
        ) from exc
    return fal_client


class FalMediaModel:
    """One fal.ai model endpoint. `endpoint` is fal's own model path
    (e.g. `"fal-ai/flux/schnell"`) — kel does not maintain a list of
    known endpoints; fal hosts hundreds of first-party and community
    models and adds more independently of kel's release cycle."""

    def __init__(self, endpoint: str, *, api_key: str | None = None, client: _FalClient | _AsyncFalClient | None = None):
        self.endpoint = endpoint
        self._api_key = api_key
        self._client = client

    def generate(self, **arguments: Any) -> MediaResult:
        with get_tracer().span("media.generate", provider="fal", endpoint=self.endpoint):
            raw = self._run_sync(arguments)
        return MediaResult(raw)

    async def agenerate(self, **arguments: Any) -> MediaResult:
        with get_tracer().span("media.generate", provider="fal", endpoint=self.endpoint):
            raw = await self._run_async(arguments)
        return MediaResult(raw)

    def _run_sync(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._client.run(self.endpoint, arguments=arguments)  # type: ignore[return-value]
        fal_client = _import_fal_client()
        if self._api_key:
            return fal_client.SyncClient(key=self._api_key).run(self.endpoint, arguments=arguments)
        return fal_client.run(self.endpoint, arguments=arguments)

    async def _run_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return await self._client.run(self.endpoint, arguments=arguments)  # type: ignore[misc]
        fal_client = _import_fal_client()
        if self._api_key:
            return await fal_client.AsyncClient(key=self._api_key).run(self.endpoint, arguments=arguments)
        return await fal_client.run_async(self.endpoint, arguments=arguments)
