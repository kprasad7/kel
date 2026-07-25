"""Wraps any ChatModel and charges every call against a shared
BudgetTracker — same decorator pattern as InstrumentedChatModel, so budget
and observability compose instead of duplicating call-wrapping logic."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from kel.budget.pricing import estimate_cost_usd
from kel.budget.tracker import BudgetTracker
from kel.models.base import ChatModel
from kel.models.types import Message, MessageStop, ModelResponse, StreamEvent


class BudgetedChatModel(ChatModel):
    def __init__(self, wrapped: ChatModel, tracker: BudgetTracker):
        self._wrapped = wrapped
        self.provider = wrapped.provider
        self.model_id = wrapped.model_id
        self.tracker = tracker

    @property
    def wrapped(self) -> ChatModel:
        return self._wrapped

    def generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        self.tracker.check_wall_time()
        response = self._wrapped.generate(messages, **kwargs)
        cost = estimate_cost_usd(self.provider, self.model_id, response.usage)
        self.tracker.record_usage(response.usage, cost)
        return response

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[StreamEvent]:
        self.tracker.check_wall_time()
        for event in self._wrapped.stream(messages, **kwargs):
            if isinstance(event, MessageStop):
                cost = estimate_cost_usd(self.provider, self.model_id, event.response.usage)
                self.tracker.record_usage(event.response.usage, cost)
            yield event

    async def agenerate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        self.tracker.check_wall_time()
        response = await self._wrapped.agenerate(messages, **kwargs)
        cost = estimate_cost_usd(self.provider, self.model_id, response.usage)
        self.tracker.record_usage(response.usage, cost)
        return response

    async def astream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.tracker.check_wall_time()
        async for event in self._wrapped.astream(messages, **kwargs):
            if isinstance(event, MessageStop):
                cost = estimate_cost_usd(self.provider, self.model_id, event.response.usage)
                self.tracker.record_usage(event.response.usage, cost)
            yield event
