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
`.run_async(...)` / `.submit(...)`) to test against a fake instead of the
real network.

Three things real-world fal.ai usage (per fal's own docs and reported
user experience) makes worth solving here rather than leaving to a bare
`fal_client.run()` call:

1. **Vendor exceptions aren't kel's own error hierarchy.** `generate()`/
   `agenerate()` translate whatever the client raises into
   `kel.models.errors.ProviderError`/`AuthenticationError`/`RateLimitError`
   (best-effort, based on an HTTP status code if the exception carries
   one — fal's own exception class names aren't hardcoded here, since
   they aren't confidently known without a live integration test).
2. **Video/long-running generations can take minutes, not seconds** —
   fal's own docs recommend the queue (`submit`/poll or webhook) over a
   blocking synchronous call for exactly this reason. `submit()` /
   `asubmit()` expose that path instead of only the blocking `generate()`.
3. **Surprise bills are a commonly reported complaint** (fal — and every
   image/video generation platform, not fal specifically — has no fixed
   per-token pricing table the way chat models do; cost varies per model,
   resolution, and duration, and a single video call can be expensive).
   Pass `budget=` a `kel.budget.BudgetTracker` plus either:
   - `cost_estimator=` (a `Callable[[dict], float]` estimating cost from
     the *arguments*, checked and reserved **before** the real network
     call) — this is the one that actually protects you, since it can
     refuse to spend money at all once the budget's exhausted, instead of
     finding out after the (possibly expensive) call already completed.
   - or `cost_usd=` (a fixed float, or a `Callable[[MediaResult], float]`
     using the actual response) for a post-hoc running total when you'd
     rather charge exact cost than an estimate.
   Passing both is redundant, not double-charged: `cost_estimator`, if
   given, takes over budget charging entirely for that call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from kel.media.errors import translate_error
from kel.media.types import MediaResult
from kel.models.types import Usage
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


class FalJobHandle:
    """A submitted, not-necessarily-complete fal.ai job — the queue-based
    flow fal's own docs recommend for slow generations (video especially:
    it can take minutes, well past what's reasonable to block a request
    thread on). Wraps whatever handle object `submit()` returned, from
    either the injected fake client or the real `fal_client` SDK.

    `result_method` defaults to `"get"` (fal_client's documented queue
    handle shape) — override it if a specific fal-client version names
    it differently."""

    def __init__(self, raw_handle: Any, *, result_method: str = "get"):
        self._raw = raw_handle
        self._result_method = result_method

    def status(self) -> Any:
        return self._raw.status()

    def result(self) -> MediaResult:
        try:
            raw = getattr(self._raw, self._result_method)()
        except Exception as exc:
            raise translate_error(exc, provider="fal") from exc
        return MediaResult(raw)

    def cancel(self) -> None:
        self._raw.cancel()


class FalMediaModel:
    """One fal.ai model endpoint. `endpoint` is fal's own model path
    (e.g. `"fal-ai/flux/schnell"`) — kel does not maintain a list of
    known endpoints; fal hosts hundreds of first-party and community
    models and adds more independently of kel's release cycle."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        client: _FalClient | _AsyncFalClient | None = None,
        budget: Any = None,
        cost_usd: float | Callable[[MediaResult], float] | None = None,
        cost_estimator: Callable[[dict[str, Any]], float] | None = None,
    ):
        self.endpoint = endpoint
        self._api_key = api_key
        self._client = client
        self._budget = budget
        self._cost_usd = cost_usd
        self._cost_estimator = cost_estimator

    def generate(self, **arguments: Any) -> MediaResult:
        self._reserve_budget(arguments)
        with get_tracer().span("media.generate", provider="fal", endpoint=self.endpoint):
            try:
                raw = self._run_sync(arguments)
            except Exception as exc:
                raise translate_error(exc, provider="fal") from exc
        result = MediaResult(raw)
        if self._cost_estimator is None:
            self._charge_budget(result)
        return result

    async def agenerate(self, **arguments: Any) -> MediaResult:
        self._reserve_budget(arguments)
        with get_tracer().span("media.generate", provider="fal", endpoint=self.endpoint):
            try:
                raw = await self._run_async(arguments)
            except Exception as exc:
                raise translate_error(exc, provider="fal") from exc
        result = MediaResult(raw)
        if self._cost_estimator is None:
            self._charge_budget(result)
        return result

    def submit(self, **arguments: Any) -> FalJobHandle:
        """Submit without waiting — poll `.status()`/`.result()` later.
        The path fal's own docs recommend for anything slow (video)
        instead of blocking on `generate()`."""
        self._reserve_budget(arguments)
        with get_tracer().span("media.submit", provider="fal", endpoint=self.endpoint):
            try:
                if self._client is not None:
                    raw_handle = self._client.submit(self.endpoint, arguments=arguments)  # type: ignore[union-attr]
                else:
                    fal_client = _import_fal_client()
                    submitter = fal_client.SyncClient(key=self._api_key) if self._api_key else fal_client
                    raw_handle = submitter.submit(self.endpoint, arguments=arguments)
            except Exception as exc:
                raise translate_error(exc, provider="fal") from exc
        return FalJobHandle(raw_handle)

    def _reserve_budget(self, arguments: dict[str, Any]) -> None:
        # Charged *before* the network call, from an estimate over the
        # request arguments — not exact, but it's the only way to refuse
        # to spend money at all once a budget's exhausted, rather than
        # discovering that after an already-expensive call completed.
        # Same "conservative, not exact" tradeoff kel.ratelimit already
        # documents for its own up-front token reservation.
        if self._budget is None or self._cost_estimator is None:
            return
        estimated = self._cost_estimator(arguments)
        self._budget.record_usage(Usage(), cost_usd=estimated)

    def _charge_budget(self, result: MediaResult) -> None:
        if self._budget is None:
            return
        cost = self._cost_usd(result) if callable(self._cost_usd) else (self._cost_usd or 0.0)
        self._budget.record_usage(Usage(), cost_usd=cost)

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
