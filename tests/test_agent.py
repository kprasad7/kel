import threading
import time

import pytest

from helpers import ScriptedModel
from kel.agents import Agent, Tool
from kel.agents.errors import EmptyModelResponseError
from kel.context import Loop, LoopBudgetExceededError, StuckLoopError
from kel.models.base import ChatModel
from kel.models.types import ModelResponse, TextPart, ToolUsePart, Usage


def _response(content, stop_reason, rid="r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=content, stop_reason=stop_reason, usage=Usage())


def test_agent_executes_tool_call_and_returns_final_response():
    tool_call = ToolUsePart(id="1", name="add", input={"a": 2, "b": 3})
    model = ScriptedModel(
        "fake-1",
        [
            _response([tool_call], "tool_use", "r1"),
            _response([TextPart(text="The sum is 5")], "end_turn", "r2"),
        ],
    )
    tool = Tool(name="add", description="add two numbers", input_schema={"type": "object"}, fn=lambda i: str(i["a"] + i["b"]))
    agent = Agent("calc", model, tools=[tool])

    response = agent.run("what is 2+3?")

    assert response.text == "The sum is 5"
    assert len(agent.memory.working.messages) == 4  # user, assistant(tool_use), user(tool_result), assistant(final)


def test_agent_reports_error_for_unknown_tool_and_continues():
    tool_call = ToolUsePart(id="1", name="mystery", input={})
    model = ScriptedModel(
        "fake-1",
        [_response([tool_call], "tool_use", "r1"), _response([TextPart(text="ok")], "end_turn", "r2")],
    )
    agent = Agent("a", model)  # no tools registered

    response = agent.run("do something")

    assert response.text == "ok"
    tool_result_part = agent.memory.working.messages[2].content[0]
    assert tool_result_part.is_error is True
    assert "unknown tool" in tool_result_part.content


def test_agent_raises_stuck_loop_error_on_repeated_identical_tool_calls():
    same_call = ToolUsePart(id="1", name="search", input={"q": "x"})
    model = ScriptedModel("fake-1", [_response([same_call], "tool_use", f"r{i}") for i in range(10)])
    tool = Tool(name="search", description="", input_schema={}, fn=lambda i: "result")
    agent = Agent("a", model, tools=[tool])

    with pytest.raises(StuckLoopError):
        agent.run("search repeatedly")


def test_agent_raises_loop_budget_exceeded_when_iterations_run_out():
    responses = [
        _response([ToolUsePart(id=str(i), name="search", input={"q": i})], "tool_use", f"r{i}") for i in range(5)
    ]
    model = ScriptedModel("fake-1", responses)
    tool = Tool(name="search", description="", input_schema={}, fn=lambda i: "result")
    agent = Agent(
        "a", model, tools=[tool], loop_factory=lambda: Loop(max_iterations=2, stuck_window=10, stuck_threshold=10)
    )

    with pytest.raises(LoopBudgetExceededError):
        agent.run("search varying")


def test_agent_forwards_max_tokens_and_temperature_to_every_generate_call():
    model = ScriptedModel("fake-1", [_response([TextPart(text="ok")], "end_turn")])
    agent = Agent("a", model, max_tokens=256, temperature=0.2)

    agent.run("hi")

    _, kwargs = model.calls[0]
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.2


def test_agent_does_not_override_provider_default_when_unset():
    model = ScriptedModel("fake-1", [_response([TextPart(text="ok")], "end_turn")])
    agent = Agent("a", model)

    agent.run("hi")

    _, kwargs = model.calls[0]
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_agent_raises_on_empty_response_and_does_not_store_it_in_memory():
    model = ScriptedModel("fake-1", [_response([], "end_turn")])
    agent = Agent("a", model)

    with pytest.raises(EmptyModelResponseError):
        agent.run("hi")

    # only the user turn should have been recorded — the empty assistant
    # turn must never make it into memory to poison later questions
    assert len(agent.memory.working.messages) == 1
    assert agent.memory.working.messages[0].role == "user"


def test_agent_allows_empty_content_when_it_is_a_legitimate_tool_use_turn():
    # a tool_use turn can legitimately have only ToolUsePart entries and no
    # text — that's not the degenerate case this guard targets
    tool_call = ToolUsePart(id="1", name="add", input={"a": 1, "b": 1})
    model = ScriptedModel(
        "fake-1",
        [
            _response([tool_call], "tool_use", "r1"),
            _response([TextPart(text="2")], "end_turn", "r2"),
        ],
    )
    tool = Tool(name="add", description="", input_schema={}, fn=lambda i: str(i["a"] + i["b"]))
    agent = Agent("a", model, tools=[tool])

    response = agent.run("what is 1+1?")

    assert response.text == "2"


def test_agent_run_executes_multiple_tool_calls_in_one_turn_concurrently():
    # regression: _execute_tool_call used to run in a plain for loop even
    # when a model requested several tools in one turn, serializing what
    # should be independent operations. Assert concurrency via event
    # ordering (slow tool starts before fast tool finishes), not a tight
    # wall-clock threshold — timing thresholds are flaky on loaded CI
    # runners (see test_realtime.py's fix for the same lesson).
    fast_started_at = None
    slow_started_at = None
    fast_finished_at = None

    def fast(_input):
        nonlocal fast_started_at, fast_finished_at
        fast_started_at = time.monotonic()
        time.sleep(0.05)
        fast_finished_at = time.monotonic()
        return "fast done"

    def slow(_input):
        nonlocal slow_started_at
        slow_started_at = time.monotonic()
        time.sleep(0.15)
        return "slow done"

    calls = [
        ToolUsePart(id="1", name="fast", input={}),
        ToolUsePart(id="2", name="slow", input={}),
    ]
    model = ScriptedModel(
        "fake-1",
        [_response(calls, "tool_use", "r1"), _response([TextPart(text="done")], "end_turn", "r2")],
    )
    tools = [
        Tool(name="fast", description="", input_schema={}, fn=fast),
        Tool(name="slow", description="", input_schema={}, fn=slow),
    ]
    agent = Agent("a", model, tools=tools)

    agent.run("do both")

    assert slow_started_at is not None and fast_started_at is not None and fast_finished_at is not None
    # if run sequentially, slow wouldn't start until fast (0.05s) finished;
    # concurrent execution means slow starts essentially immediately,
    # before fast's sleep(0.05) has had time to complete
    assert slow_started_at < fast_finished_at


async def test_agent_arun_executes_multiple_tool_calls_in_one_turn_concurrently():
    fast_started_at = None
    slow_started_at = None
    fast_finished_at = None

    def fast(_input):
        nonlocal fast_started_at, fast_finished_at
        fast_started_at = time.monotonic()
        time.sleep(0.05)
        fast_finished_at = time.monotonic()
        return "fast done"

    def slow(_input):
        nonlocal slow_started_at
        slow_started_at = time.monotonic()
        time.sleep(0.15)
        return "slow done"

    calls = [
        ToolUsePart(id="1", name="fast", input={}),
        ToolUsePart(id="2", name="slow", input={}),
    ]
    model = ScriptedModel(
        "fake-1",
        [_response(calls, "tool_use", "r1"), _response([TextPart(text="done")], "end_turn", "r2")],
    )
    tools = [
        Tool(name="fast", description="", input_schema={}, fn=fast),
        Tool(name="slow", description="", input_schema={}, fn=slow),
    ]
    agent = Agent("a", model, tools=tools)

    await agent.arun("do both")

    assert slow_started_at is not None and fast_started_at is not None and fast_finished_at is not None
    assert slow_started_at < fast_finished_at


def test_agent_rejects_tool_call_when_approval_hook_returns_false():
    tool_call = ToolUsePart(id="1", name="delete_file", input={"path": "/etc/passwd"})
    model = ScriptedModel(
        "fake-1",
        [_response([tool_call], "tool_use", "r1"), _response([TextPart(text="ok")], "end_turn", "r2")],
    )
    tool_called = False

    def delete_file(_input):
        nonlocal tool_called
        tool_called = True
        return "deleted"

    tool = Tool(name="delete_file", description="", input_schema={}, fn=delete_file)
    agent = Agent("a", model, tools=[tool], approve_tool_call=lambda name, input: False)

    agent.run("delete a file")

    assert tool_called is False
    tool_result = agent.memory.working.messages[2].content[0]
    assert tool_result.is_error is True
    assert "rejected by approval hook" in tool_result.content


def test_agent_runs_tool_call_when_approval_hook_returns_true():
    tool_call = ToolUsePart(id="1", name="add", input={"a": 1, "b": 2})
    model = ScriptedModel(
        "fake-1",
        [_response([tool_call], "tool_use", "r1"), _response([TextPart(text="3")], "end_turn", "r2")],
    )
    tool = Tool(name="add", description="", input_schema={}, fn=lambda i: str(i["a"] + i["b"]))
    approvals: list[tuple[str, dict]] = []

    def approve(name, input):
        approvals.append((name, input))
        return True

    agent = Agent("a", model, tools=[tool], approve_tool_call=approve)

    response = agent.run("what is 1+2?")

    assert response.text == "3"
    assert approvals == [("add", {"a": 1, "b": 2})]


def test_agent_with_no_approval_hook_runs_every_tool_call_as_before():
    tool_call = ToolUsePart(id="1", name="add", input={"a": 1, "b": 1})
    model = ScriptedModel(
        "fake-1",
        [_response([tool_call], "tool_use", "r1"), _response([TextPart(text="2")], "end_turn", "r2")],
    )
    tool = Tool(name="add", description="", input_schema={}, fn=lambda i: str(i["a"] + i["b"]))
    agent = Agent("a", model, tools=[tool])

    response = agent.run("what is 1+1?")

    assert response.text == "2"


class _ConcurrencyTrackingModel(ChatModel):
    """Records the peak number of generate() calls actually in flight at
    once — the invariant an Agent-level lock must enforce: at most 1,
    never truly concurrent, no matter how many threads call run() on the
    same Agent instance at once."""

    provider = "fake"
    model_id = "fake"

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def generate(self, messages, **kwargs):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(0.02)
        with self._lock:
            self._in_flight -= 1
        return _response([TextPart(text="ok")], "end_turn")

    def stream(self, messages, **kwargs):
        raise NotImplementedError


def test_agent_serializes_concurrent_run_calls_on_the_same_instance():
    # regression: concurrent run() calls on one shared Agent (the pattern
    # kel.sdk.serve/serve_websocket/fastapi_adapter's "one Agent handles
    # every request" naturally invites) used to interleave their
    # memory.remember_turn() writes with no ordering guarantee —
    # reproduced as 5 concurrent calls producing 5 user messages
    # back-to-back, then 5 assistant messages, instead of 5 alternating
    # pairs. The Agent-level lock must prevent two calls from ever
    # actually being in the generate() critical section at once.
    model = _ConcurrencyTrackingModel()
    agent = Agent("shared", model)

    threads = [threading.Thread(target=agent.run, args=(f"question-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.max_in_flight == 1
    roles = [m.role for m in agent.memory.working.messages]
    assert roles == ["user", "assistant"] * 5


