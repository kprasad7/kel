"""Shared test doubles."""

from __future__ import annotations

from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse


class ScriptedModel(ChatModel):
    """Returns pre-scripted responses in order, one per generate() call.
    Records every call's messages/kwargs so tests can assert on what the
    agent actually sent (e.g. "did agent B see agent A's output")."""

    provider = "fake"

    def __init__(self, model_id: str, responses: list[ModelResponse]):
        self.model_id = model_id
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], dict]] = []

    def generate(self, messages: list[Message], **kwargs) -> ModelResponse:
        self.calls.append((list(messages), kwargs))
        if not self._responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self._responses.pop(0)

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages: list[Message], **kwargs) -> ModelResponse:
        return self.generate(messages, **kwargs)


class ScriptedStreamModel(ChatModel):
    """Like ScriptedModel, but for stream()/astream(): each entry in
    `event_lists` is the full sequence of StreamEvents yielded by one
    stream() call (must end with a MessageStop, same as a real adapter)."""

    provider = "fake"

    def __init__(self, model_id: str, event_lists: list[list]):
        self.model_id = model_id
        self._event_lists = list(event_lists)
        self.calls: list[tuple[list[Message], dict]] = []

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    def stream(self, messages: list[Message], **kwargs):
        self.calls.append((list(messages), kwargs))
        if not self._event_lists:
            raise AssertionError("ScriptedStreamModel ran out of scripted event lists")
        yield from self._event_lists.pop(0)

    async def astream(self, messages: list[Message], **kwargs):
        self.calls.append((list(messages), kwargs))
        if not self._event_lists:
            raise AssertionError("ScriptedStreamModel ran out of scripted event lists")
        for event in self._event_lists.pop(0):
            yield event
