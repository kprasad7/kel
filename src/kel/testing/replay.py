"""Replays a Cassette's recorded responses deterministically — no network
calls, no API key needed, same responses every CI run.

v1 replays strictly in recorded order (call N gets interaction N). That
matches the common case: a fixed test scenario that calls the model the
same number of times in the same order every run. Matching by request
content instead of position is a reasonable future upgrade for tests that
don't call the model in a fixed sequence — not built here."""

from __future__ import annotations

from typing import Any

from kel.models.base import ChatModel
from kel.models.errors import KelError
from kel.models.types import Message, ModelResponse
from kel.testing.cassette import Cassette


class ReplayExhaustedError(KelError):
    pass


class ReplayChatModel(ChatModel):
    def __init__(self, cassette: Cassette, *, provider: str = "replay", model_id: str = "replay"):
        self.provider = provider
        self.model_id = model_id
        self.cassette = cassette
        self._index = 0

    def generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        if self._index >= len(self.cassette.interactions):
            raise ReplayExhaustedError(
                f"cassette exhausted after {self._index} interaction(s); the code under test "
                "called generate() more times than were recorded"
            )
        interaction = self.cassette.interactions[self._index]
        self._index += 1
        return interaction.response

    def stream(self, messages: list[Message], **kwargs: Any):
        raise NotImplementedError("ReplayChatModel only supports generate() in v1")
