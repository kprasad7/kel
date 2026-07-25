"""Orchestration-only, per DESIGN.md §7's descope: kel does not build its
own speech/media infrastructure (that's LiveKit/Daily/Deepgram/Cartesia/
ElevenLabs/Tavus territory) — it defines the seam a real vendor SDK plugs
into. **No concrete STT/TTS implementations ship here.** Wire up a real
vendor behind these two Protocols yourself; kel.realtime's actual
contribution is the dual-path orchestration pattern in dual_path.py, which
works with any provider that satisfies this interface."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class STTResult(BaseModel):
    text: str
    is_final: bool = True


class STTProvider(Protocol):
    def transcribe(self, audio: bytes) -> STTResult: ...


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> bytes: ...
