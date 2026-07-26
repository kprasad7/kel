"""Adapter for Replicate model endpoints (`pip install kel[replicate]`) —
a second `kel.media` provider alongside fal, proving the media-gateway
abstraction (§16 of USAGE.md) isn't just "fal wrapped": image, video, and
audio generation on Replicate follow the same run(model, input) shape fal
uses for its own endpoints, just under a different vendor SDK.

`model_ref` is Replicate's own reference — `"owner/model"` resolves to
that model's *latest* version automatically (convenient, but the exact
model run can drift over time as the owner pushes new versions); pin
`"owner/model:version"` for anything that needs to reproduce exactly.
kel does not maintain a list of Replicate's models — it hosts thousands
of first-party and community ones, added independently of kel's release
cycle, same as fal.

Replicate's `run()` output shape varies by model: a single URL/file-like
object, a list of them, or (for some structured-output models) a dict
already. `_normalize_output` folds the first two into the same
`{"url": ...}`-keyed shape `kel.media.MediaResult`'s extraction already
understands, rather than teaching `MediaResult` about Replicate
specifically.

Same dependency-injection shape as `FalMediaModel`: pass `client=`
(anything exposing `.run(model_ref, input=...)` / an async counterpart /
`.predictions.create(...)`) to test against a fake instead of the real
network. Also carries the same three fixes `FalMediaModel` does — error
translation, a queue-based `submit()` for slow generations, and
`budget=`/`cost_usd=`/`cost_estimator=` for cost governance — since none
of those are fal-specific problems.

Less confidently verified than `FalMediaModel`'s `fal_client` shape: the
exact `predictions.create()` queue API is implemented against Replicate's
documented pattern but not exercised against a live account. `generate()`/
`agenerate()` (the plain `run()`/`async_run()` path) are the better-known,
lower-risk surface if you don't need `submit()`'s queue/poll behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from kel.media.errors import translate_error
from kel.media.types import MediaResult
from kel.models.types import Usage
from kel.observability import get_tracer


class _ReplicateClient(Protocol):
    def run(self, model_ref: str, input: dict[str, Any]) -> Any: ...


def _import_replicate() -> Any:
    try:
        import replicate
    except ImportError as exc:
        raise ImportError(
            "The replicate package is required for Replicate media generation. "
            "Install it with `pip install kel[replicate]`."
        ) from exc
    return replicate


def _normalize_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, list):
        return {"output": [{"url": str(item)} for item in output]}
    return {"output": {"url": str(output)}}


class ReplicateJobHandle:
    """A submitted, not-necessarily-complete Replicate prediction —
    Replicate's own queue path (`predictions.create` + poll/wait) for
    slow generations (video) instead of blocking on `generate()`. Wraps
    whatever prediction object `submit()` returned."""

    def __init__(self, raw_prediction: Any):
        self._raw = raw_prediction

    def status(self) -> Any:
        reload = getattr(self._raw, "reload", None)
        if callable(reload):
            reload()
        return getattr(self._raw, "status", None)

    def result(self) -> MediaResult:
        try:
            wait = getattr(self._raw, "wait", None)
            if callable(wait):
                wait()
            output = self._raw.output
        except Exception as exc:
            raise translate_error(exc, provider="replicate") from exc
        return MediaResult(_normalize_output(output))

    def cancel(self) -> None:
        self._raw.cancel()


class ReplicateMediaModel:
    """One Replicate model reference (`"owner/model"` or
    `"owner/model:version"`)."""

    def __init__(
        self,
        model_ref: str,
        *,
        api_key: str | None = None,
        client: _ReplicateClient | None = None,
        budget: Any = None,
        cost_usd: float | Callable[[MediaResult], float] | None = None,
        cost_estimator: Callable[[dict[str, Any]], float] | None = None,
    ):
        self.model_ref = model_ref
        self._api_key = api_key
        self._client = client
        self._budget = budget
        self._cost_usd = cost_usd
        self._cost_estimator = cost_estimator

    def generate(self, **arguments: Any) -> MediaResult:
        self._reserve_budget(arguments)
        with get_tracer().span("media.generate", provider="replicate", endpoint=self.model_ref):
            try:
                output = self._run_sync(arguments)
            except Exception as exc:
                raise translate_error(exc, provider="replicate") from exc
        result = MediaResult(_normalize_output(output))
        if self._cost_estimator is None:
            self._charge_budget(result)
        return result

    async def agenerate(self, **arguments: Any) -> MediaResult:
        self._reserve_budget(arguments)
        with get_tracer().span("media.generate", provider="replicate", endpoint=self.model_ref):
            try:
                output = await self._run_async(arguments)
            except Exception as exc:
                raise translate_error(exc, provider="replicate") from exc
        result = MediaResult(_normalize_output(output))
        if self._cost_estimator is None:
            self._charge_budget(result)
        return result

    def submit(self, **arguments: Any) -> ReplicateJobHandle:
        """Submit without waiting — poll `.status()`/`.result()` later,
        via Replicate's own prediction-queue API."""
        self._reserve_budget(arguments)
        with get_tracer().span("media.submit", provider="replicate", endpoint=self.model_ref):
            try:
                if self._client is not None:
                    predictions = getattr(self._client, "predictions", None)
                    if predictions is not None:
                        raw_prediction = predictions.create(model=self.model_ref, input=arguments)
                    else:
                        raw_prediction = self._client.run(self.model_ref, input=arguments)
                else:
                    replicate = _import_replicate()
                    r = replicate.Client(api_token=self._api_key) if self._api_key else replicate
                    raw_prediction = r.predictions.create(model=self.model_ref, input=arguments)
            except Exception as exc:
                raise translate_error(exc, provider="replicate") from exc
        return ReplicateJobHandle(raw_prediction)

    def _reserve_budget(self, arguments: dict[str, Any]) -> None:
        if self._budget is None or self._cost_estimator is None:
            return
        estimated = self._cost_estimator(arguments)
        self._budget.record_usage(Usage(), cost_usd=estimated)

    def _charge_budget(self, result: MediaResult) -> None:
        if self._budget is None:
            return
        cost = self._cost_usd(result) if callable(self._cost_usd) else (self._cost_usd or 0.0)
        self._budget.record_usage(Usage(), cost_usd=cost)

    def _run_sync(self, arguments: dict[str, Any]) -> Any:
        if self._client is not None:
            return self._client.run(self.model_ref, input=arguments)
        replicate = _import_replicate()
        if self._api_key:
            return replicate.Client(api_token=self._api_key).run(self.model_ref, input=arguments)
        return replicate.run(self.model_ref, input=arguments)

    async def _run_async(self, arguments: dict[str, Any]) -> Any:
        if self._client is not None:
            run_async = getattr(self._client, "async_run", None)
            if run_async is None:
                raise NotImplementedError("injected client has no async_run method")
            return await run_async(self.model_ref, input=arguments)
        replicate = _import_replicate()
        if self._api_key:
            return await replicate.Client(api_token=self._api_key).async_run(self.model_ref, input=arguments)
        return await replicate.async_run(self.model_ref, input=arguments)
