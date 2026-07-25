import pytest

from helpers import ScriptedModel
from kel.agents import Agent, Tool
from kel.context import Loop, LoopBudgetExceededError, StuckLoopError
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
