"""Single-agent tool-calling loop, built directly on kel.context.Loop
(step budget + stuck-loop detection) and kel.memory.Memory (working +
episodic). This is the building block kel.agents' multi-agent orchestration
patterns compose — a "multi-agent framework" that can't run one agent
well isn't worth having a supervisor pattern on top of."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from kel.agents.errors import EmptyModelResponseError
from kel.agents.events import ToolResultEvent
from kel.agents.tool import Tool
from kel.context.loop import Loop
from kel.memory.memory import Memory
from kel.models.base import ChatModel
from kel.models.types import Message, MessageStop, ModelResponse, Role, StreamEvent, ToolResultPart, ToolUsePart

ApprovalHook = Callable[[str, dict[str, Any]], bool]


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
        approve_tool_call: ApprovalHook | None = None,
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
        # Injected gate, not a hardcoded mechanism — same DI shape as
        # `model`/`tools`/`memory`/`loop_factory` above. Given `(name,
        # input)`, return True to let the call proceed or False to reject
        # it before it ever runs. Checked once, in `_execute_tool_call`,
        # which every run variant (run/arun/run_stream/arun_stream) and
        # the concurrent tool-dispatch helpers all funnel through — so
        # one hook covers every path without duplicating the check.
        # Unset means every tool call is approved, matching prior behavior.
        self.approve_tool_call = approve_tool_call
        # One Agent instance = one conversation. Concurrent calls to
        # run()/run_stream() (or arun()/arun_stream()) on the *same*
        # instance would otherwise interleave their memory.remember_turn()
        # writes with no ordering guarantee — e.g. two concurrent run()
        # calls from a server handling two requests against one shared
        # Agent can scramble both turns' user/assistant messages together
        # (reproduced: 5 concurrent calls produced 5 user messages back
        # to back, then 5 assistant messages, not alternating pairs).
        # These locks serialize calls on one instance so a turn's memory
        # writes always complete atomically relative to any other call on
        # the same Agent — construct one Agent per session/connection for
        # real concurrent multi-user serving (kel.sdk.serve_websocket/
        # fastapi_adapter) rather than sharing one Agent across callers.
        # Sync and async paths use separate locks (a thread lock can't be
        # awaited); mixing run()/arun() calls on one instance from
        # different threads concurrently is not covered by this and
        # should be avoided.
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

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
        with self._sync_lock:
            return self._run_locked(user_input)

    def _run_locked(self, user_input: str) -> ModelResponse:
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

            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
            result_parts: list[Any] = self._execute_tool_calls(response.tool_calls)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    async def arun(self, user_input: str) -> ModelResponse:
        """Async equivalent of `run()` — same loop, uses the model's real
        `agenerate()` instead of blocking `generate()`."""
        async with self._async_lock:
            return await self._arun_locked(user_input)

    async def _arun_locked(self, user_input: str) -> ModelResponse:
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

            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)
            result_parts: list[Any] = await self._aexecute_tool_calls(response.tool_calls)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    def run_stream(self, user_input: str) -> Iterator[StreamEvent | ToolResultEvent]:
        """Like `run()`, but yields StreamEvents as they arrive across
        every model call in the loop — including intermediate tool-calling
        turns, not just the final answer. A `ToolResultEvent` is yielded
        after each tool finishes, so a UI can show "calling search..."
        progress that model-level streaming alone can't express."""
        with self._sync_lock:
            yield from self._run_stream_locked(user_input)

    def _run_stream_locked(self, user_input: str) -> Iterator[StreamEvent | ToolResultEvent]:
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

            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)

            result_parts: list[Any] = []
            for call, result_part in self._execute_tool_calls_streaming(response.tool_calls):
                result_parts.append(result_part)
                yield ToolResultEvent(tool_use_id=call.id, name=call.name, result=result_part)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    async def arun_stream(self, user_input: str) -> AsyncIterator[StreamEvent | ToolResultEvent]:
        """Async equivalent of `run_stream()`, using the model's real `astream()`."""
        async with self._async_lock:
            async for event in self._arun_stream_locked(user_input):
                yield event

    async def _arun_stream_locked(self, user_input: str) -> AsyncIterator[StreamEvent | ToolResultEvent]:
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

            for call in response.tool_calls:
                signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
                loop.record_action(signature)

            result_parts: list[Any] = []
            async for call, result_part in self._aexecute_tool_calls_streaming(response.tool_calls):
                result_parts.append(result_part)
                yield ToolResultEvent(tool_use_id=call.id, name=call.name, result=result_part)

            self.memory.remember_turn(Message(role=Role.USER, content=result_parts))

    def _execute_tool_calls(self, tool_calls: list[ToolUsePart]) -> list[ToolResultPart]:
        """Runs every tool call from one model turn concurrently instead
        of one at a time — a model requesting N tools in a single turn is
        asking for N independent operations (e.g. N web searches across
        different sources), and serializing them needlessly adds up their
        latencies instead of taking the max. Same `ThreadPoolExecutor`
        concurrency primitive `kel.agents.orchestration.run_parallel`
        already uses at the multi-agent level, just applied at the
        single-agent tool-dispatch level too. Order of the returned list
        matches `tool_calls`' order — each `ToolResultPart` also carries
        its own `tool_use_id`, so callers never depend on list order, but
        deterministic order is simpler to reason about than completion
        order for the non-streaming (`run`/`arun`) callers."""
        if len(tool_calls) == 1:
            call = tool_calls[0]
            return [self._execute_tool_call(call.id, call.name, call.input)]
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = [pool.submit(self._execute_tool_call, call.id, call.name, call.input) for call in tool_calls]
            return [f.result() for f in futures]

    def _execute_tool_calls_streaming(
        self, tool_calls: list[ToolUsePart]
    ) -> Iterator[tuple[ToolUsePart, ToolResultPart]]:
        """Same concurrency as `_execute_tool_calls`, but yields each
        `(call, result)` pair as soon as that specific call finishes
        (`as_completed`, not submission order) — `run_stream`'s whole
        point is surfacing per-tool progress to a UI, and reporting
        completions in real order is strictly more useful than waiting
        for the slowest call before reporting any of them."""
        if len(tool_calls) == 1:
            call = tool_calls[0]
            yield call, self._execute_tool_call(call.id, call.name, call.input)
            return
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            future_to_call = {
                pool.submit(self._execute_tool_call, call.id, call.name, call.input): call for call in tool_calls
            }
            for future in as_completed(future_to_call):
                yield future_to_call[future], future.result()

    async def _aexecute_tool_call(self, call_id: str, name: str, tool_input: dict[str, Any]) -> ToolResultPart:
        # Tool.fn is a plain sync callable — run it in a thread so it
        # can't block the event loop, same as any other sync I/O call
        # made from async code.
        return await asyncio.to_thread(self._execute_tool_call, call_id, name, tool_input)

    async def _aexecute_tool_calls(self, tool_calls: list[ToolUsePart]) -> list[ToolResultPart]:
        return list(
            await asyncio.gather(*(self._aexecute_tool_call(c.id, c.name, c.input) for c in tool_calls))
        )

    async def _aexecute_tool_calls_streaming(
        self, tool_calls: list[ToolUsePart]
    ) -> AsyncIterator[tuple[ToolUsePart, ToolResultPart]]:
        async def _run_and_tag(call: ToolUsePart) -> tuple[ToolUsePart, ToolResultPart]:
            return call, await self._aexecute_tool_call(call.id, call.name, call.input)

        for coro in asyncio.as_completed([_run_and_tag(call) for call in tool_calls]):
            yield await coro

    def _execute_tool_call(self, call_id: str, name: str, tool_input: dict[str, Any]) -> ToolResultPart:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResultPart(tool_use_id=call_id, content=f"error: unknown tool {name!r}", is_error=True)
        if self.approve_tool_call is not None and not self.approve_tool_call(name, tool_input):
            return ToolResultPart(
                tool_use_id=call_id, content=f"error: tool call {name!r} rejected by approval hook", is_error=True
            )
        try:
            return ToolResultPart(tool_use_id=call_id, content=tool(tool_input), is_error=False)
        except Exception as exc:
            return ToolResultPart(tool_use_id=call_id, content=f"error: {exc}", is_error=True)
