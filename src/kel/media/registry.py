"""`"provider:model_ref"` spec -> media model factory, the same registry
shape `kel.models.registry` uses for chat models — a small core-maintained
set (today: `fal`, since fal.ai's one generic endpoint-submission API
already covers image/video/audio/lipsync) with the registry open for
another vendor to be added the same way, without kel's core growing an
ever-longer if/elif chain."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.media.fal import FalMediaModel

_MediaModelFactory = Callable[..., Any]

_PROVIDERS: dict[str, _MediaModelFactory] = {}


def register_media_provider(prefix: str, factory: _MediaModelFactory) -> None:
    """Register a media-model factory under a provider prefix. `factory`
    is called as `factory(model_ref, **kwargs)`."""
    _PROVIDERS[prefix] = factory


def _register_builtin_fal(model_ref: str, **kwargs: Any) -> FalMediaModel:
    return FalMediaModel(model_ref, **kwargs)


register_media_provider("fal", _register_builtin_fal)


def _get_media_model(spec: str, **kwargs: Any) -> Any:
    if ":" not in spec:
        raise ValueError(f"Media model spec must be 'provider:model_ref', got {spec!r}. Known providers: {sorted(_PROVIDERS)}")
    provider, model_ref = spec.split(":", 1)
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(f"Unknown media provider {provider!r}. Known providers: {sorted(_PROVIDERS)}")
    return factory(model_ref, **kwargs)


def get_image_model(spec: str, *, api_key: str | None = None, client: Any = None) -> Any:
    """Example: `get_image_model("fal:fal-ai/flux/schnell")`. Image
    "scale" (resolution/aspect ratio) is just an argument the specific
    endpoint defines — pass it to `.generate(**arguments)`, not here."""
    return _get_media_model(spec, api_key=api_key, client=client)


def get_video_model(spec: str, *, api_key: str | None = None, client: Any = None) -> Any:
    """Example: `get_video_model("fal:fal-ai/kling-video/v1.6/standard/text-to-video")`."""
    return _get_media_model(spec, api_key=api_key, client=client)


def get_audio_model(spec: str, *, api_key: str | None = None, client: Any = None) -> Any:
    """Example: `get_audio_model("fal:fal-ai/elevenlabs/tts/turbo-v2.5")`."""
    return _get_media_model(spec, api_key=api_key, client=client)


def get_lipsync_model(spec: str, *, api_key: str | None = None, client: Any = None) -> Any:
    """Example: `get_lipsync_model("fal:fal-ai/sync-lipsync")`."""
    return _get_media_model(spec, api_key=api_key, client=client)
