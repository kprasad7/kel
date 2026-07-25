"""Single-agent tool-calling loop, built directly on kel.context.Loop
(step budget + stuck-loop detection) and kel.memory.Memory (working +
episodic). This is the building block kel.agents' multi-agent orchestration
patterns compose — a "multi-agent framework" that can't run one agent
well isn't worth having a supervisor pattern on top of."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from kel.agents.errors import EmptyModelResponseError
from kel.agents.events import ToolResultEvent
from kel.agents.tool import Tool
from kel.context.loop import Loop
from kel.memory.memory import Memory
from kel.models.base import ChatModel
from kel.models.types import Message, MessageStop, ModelResponse, Role, StreamEvent, ToolResultPart


class Agent:
    def __init__(
        self,
        name: str,
        model: ChatModel,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        loop_factory: Callable[[], Loop] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory or Memory(session_id=name)
        self._loop_factory = loop_factory or (lambda: Loop(max_iterations=10))
        # only forwarded to the model call when explicitly set, so leaving
        # these unset preserves each provider adapter's own default (e.g.
        # max_tokens=1024) instead of silently overriding it with kel's own
        # opinion of what the default should be.
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    @staticmethod
    def _validate_response(response: ModelResponse) -> None:
        # A turn with no content at all (no text, no tool calls) that
        # isn't a tool-use turn is a truncated/degenerate response, not a
        # legitimate empty answer. Storing it into memory would leave a
        # bare `Message(role=ASSISTANT, content=[])` in the conversation
        # history — and because Agent.memory persists across every run()
        # call, that one bad turn silently corrupts every later question
        # in the same session. Fail loudly on the turn that caused it
        # instead.
        if not response.content and response.stop_reason != "tool_use":
            raise EmptyModelResponseError(
                f"model returned an empty response (stop_reason={response.stop_reason!r}) "
                "with no text and no tool calls; refusing to store it in memory",
                stop_reason=response.stop_reason,
            )

    def run(self, user_input: str) -> ModelResponse:
        self.memory.remember_turn(Message.user(user_input))
        loop = self._loop_factory()
        tool_specs = [t.to_spec() for t in self.tools.values()] or None

        while True:
            loop.step()
            response = self.model.generate(
                self.memory.working.messages,
                system=self.system_prompt,
                tools=tool_specs,
                **self._generation_kwargs(),
            )
            self._validate_response(response)
            self.memory.remember_turn(Message(role=Role.ASSISTANT, content=response.content))

            if response.stop_reason != "tool_use":
                return response

            result_parts: list[Any] = []
            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
                result_parts.append(self._execute_tool_call(call.id, call.name, call.input))

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    async def arun(self, user_input: str) -> ModelResponse:
        """Async equivalent of `run()` — same loop, uses the model's real
        `agenerate()` instead of blocking `generate()`."""
        self.memory.remember_turn(Message.user(user_input))
        loop = self._loop_factory()
        tool_specs = [t.to_spec() for t in self.tools.values()] or None

        while True:
            loop.step()
            response = await self.model.agenerate(
                self.memory.working.messages,
                system=self.system_prompt,
                tools=tool_specs,
                **self._generation_kwargs(),
            )
            self._validate_response(response)
            self.memory.remember_turn(Message(role=Role.ASSISTANT, content=response.content))

            if response.stop_reason != "tool_use":
                return response

            result_parts: list[Any] = []
            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
                result_parts.append(self._execute_tool_call(call.id, call.name, call.input))

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    def run_stream(self, user_input: str) -> Iterator[StreamEvent | ToolResultEvent]:
        """Like `run()`, but yields StreamEvents as they arrive across
        every model call in the loop — including intermediate tool-calling
        turns, not just the final answer. A `ToolResultEvent` is yielded
        after each tool finishes, so a UI can show "calling search..."
        progress that model-level streaming alone can't express."""
        self.memory.remember_turn(Message.user(user_input))
        loop = self._loop_factory()
        tool_specs = [t.to_spec() for t in self.tools.values()] or None

        while True:
            loop.step()
            response: ModelResponse | None = None
            for event in self.model.stream(
                self.memory.working.messages,
                system=self.system_prompt,
                tools=tool_specs,
                **self._generation_kwargs(),
            ):
                yield event
                if isinstance(event, MessageStop):
                    response = event.response

            assert response is not None  # every stream() implementation must end with a MessageStop
            self._validate_response(response)
            self.memory.remember_turn(Message(role=Role.ASSISTANT, content=response.content))

            if response.stop_reason != "tool_use":
                return

            result_parts: list[Any] = []
            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
                result_part = self._execute_tool_call(call.id, call.name, call.input)
                result_parts.append(result_part)
                yield ToolResultEvent(tool_use_id=call.id, name=call.name, result=result_part)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    async def arun_stream(self, user_input: str) -> AsyncIterator[StreamEvent | ToolResultEvent]:
        """Async equivalent of `run_stream()`, using the model's real `astream()`."""
        self.memory.remember_turn(Message.user(user_input))
        loop = self._loop_factory()
        tool_specs = [t.to_spec() for t in self.tools.values()] or None

        while True:
            loop.step()
            response: ModelResponse | None = None
            async for event in self.model.astream(
                self.memory.working.messages,
                system=self.system_prompt,
                tools=tool_specs,
                **self._generation_kwargs(),
            ):
                yield event
                if isinstance(event, MessageStop):
                    response = event.response

            assert response is not None
            self._validate_response(response)
            self.memory.remember_turn(Message(role=Role.ASSISTANT, content=response.content))

            if response.stop_reason != "tool_use":
                return

            result_parts: list[Any] = []
            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
                result_part = self._execute_tool_call(call.id, call.name, call.input)
                result_parts.append(result_part)
                yield ToolResultEvent(tool_use_id=call.id, name=call.name, result=result_part)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    def _execute_tool_call(self, call_id: str, name: str, tool_input: dict[str, Any]) -> ToolResultPart:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResultPart(tool_use_id=call_id, content=f"error: unknown tool {name!r}", is_error=True)
        try:
            return ToolResultPart(tool_use_id=call_id, content=tool(tool_input), is_error=False)
        except Exception as exc:
            return ToolResultPart(tool_use_id=call_id, content=f"error: {exc}", is_error=True)
