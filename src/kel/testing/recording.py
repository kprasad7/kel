from __future__ import annotations

from typing import Any

from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse
from kel.testing.cassette import Cassette, Interaction


def _serialize_request(messages: list[Message], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [m.model_dump(mode="json") for m in messages],
        "kwargs": {k: v for k, v in kwargs.items() if k != "tools"},
    }


class RecordingChatModel(ChatModel):
    """Wraps a real ChatModel; every `generate()` call is captured into a
    Cassette. v1 only supports recording `generate()`, not `stream()` —
    capturing a faithful, replayable stream-event sequence is meaningfully
    more work than replaying a final response and isn't needed for the
    common case (golden-trace regression tests on the end result)."""

    def __init__(self, wrapped: ChatModel, cassette: Cassette | None = None):
        self._wrapped = wrapped
        self.provider = wrapped.provider
        self.model_id = wrapped.model_id
        self.cassette = cassette or Cassette()

    def generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        response = self._wrapped.generate(messages, **kwargs)
        self.cassette.interactions.append(
            Interaction(request=_serialize_request(messages, kwargs), response=response)
        )
        return response

    def stream(self, messages: list[Message], **kwargs: Any):
        raise NotImplementedError("RecordingChatModel only supports generate() in v1")
