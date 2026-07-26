"""Real `kel.realtime.STTProvider`/`TTSProvider` implementations backed by
a fal.ai endpoint — closes the "no bundled STT/TTS, wire up your own
vendor" gap `kel.realtime` documents, for anyone who picked fal
specifically. Still just one adapter among many possible vendors: nothing
here is wired into `kel.realtime.run_dual_path` automatically, same
"composable, not forced" shape as the rest of kel.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from kel.media.fal import FalMediaModel
from kel.realtime.providers import STTResult

_ALLOWED_SCHEMES = {"http", "https"}


def _default_download(url: str) -> bytes:
    # same SSRF-hardening as kel.tools.web_fetch: validate the scheme
    # before ever calling urlopen, since this URL comes from a fal
    # response and shouldn't be trusted to always be a plain https link.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported media URL scheme {scheme!r} — only http/https are allowed")
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310 - scheme validated above
        return resp.read()


class FalTTSProvider:
    """Satisfies `kel.realtime.TTSProvider` (`synthesize(text) -> bytes`)
    using a fal.ai TTS endpoint. `text_arg` is configurable because fal's
    various TTS models don't all name their text input the same way."""

    def __init__(
        self,
        media_model: FalMediaModel,
        *,
        text_arg: str = "text",
        extra_arguments: dict[str, Any] | None = None,
        download: Callable[[str], bytes] | None = None,
    ):
        self._media_model = media_model
        self._text_arg = text_arg
        self._extra = extra_arguments or {}
        self._download = download or _default_download

    def synthesize(self, text: str) -> bytes:
        result = self._media_model.generate(**{self._text_arg: text}, **self._extra)
        if not result.urls:
            raise ValueError(f"fal endpoint {self._media_model.endpoint!r} returned no audio URL: {result.raw}")
        return self._download(result.urls[0])


class FalSTTProvider:
    """Satisfies `kel.realtime.STTProvider` (`transcribe(audio) -> STTResult`)
    using a fal.ai STT endpoint. fal endpoints take an audio *URL*, not
    raw bytes, so `upload_fn` (audio bytes -> a URL fal can fetch) is
    required — inject your own (fal's own file upload, S3 presigned URL,
    anything reachable over HTTP) rather than kel guessing at one vendor's
    upload API."""

    def __init__(
        self,
        media_model: FalMediaModel,
        upload_fn: Callable[[bytes], str],
        *,
        audio_arg: str = "audio_url",
        text_key: str = "text",
        extra_arguments: dict[str, Any] | None = None,
    ):
        self._media_model = media_model
        self._upload_fn = upload_fn
        self._audio_arg = audio_arg
        self._text_key = text_key
        self._extra = extra_arguments or {}

    def transcribe(self, audio: bytes) -> STTResult:
        url = self._upload_fn(audio)
        result = self._media_model.generate(**{self._audio_arg: url}, **self._extra)
        text = result.raw.get(self._text_key, "")
        return STTResult(text=text, is_final=True)
