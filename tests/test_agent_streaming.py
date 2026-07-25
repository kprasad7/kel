import pytest

from kel.agents import Agent, Tool, ToolResultEvent
from kel.models.types import MessageStop, ModelResponse, TextDelta, TextPart, ToolCallDelta, ToolUsePart, Usage
from helpers import ScriptedModel, ScriptedStreamModel


def _final_text_response(text: str, rid: str) -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())


def test_run_stream_yields_text_deltas_and_final_message_stop():
    events = [
        [TextDelta(text="Hel"), TextDelta(text="lo"), MessageStop(response=_final_text_response("Hello", "r1"))]
    ]
    model = ScriptedStreamModel("fake-1", events)
    agent = Agent("a", model)

    collected = list(agent.run_stream("hi"))

    text = "".join(e.text for e in collected if isinstance(e, TextDelta))
    assert text == "Hello"
    assert isinstance(collected[-1], MessageStop)
    assert collected[-1].response.text == "Hello"
    # memory recorded the turn
    assert agent.memory.working.messages[-1].text == "Hello"


def test_run_stream_executes_tools_and_yields_tool_result_events():
    tool_call = ToolUsePart(id="1", name="add", input={"a": 2, "b": 3})
    tool_use_response = ModelResponse(
        id="r1", model="fake-1", content=[tool_call], stop_reason="tool_use", usage=Usage()
    )
    events = [
        [ToolCallDelta(tool_call=tool_call), MessageStop(response=tool_use_response)],
        [MessageStop(response=_final_text_response("the sum is 5", "r2"))],
    ]
    model = ScriptedStreamModel("fake-1", events)
    tool = Tool(name="add", description="add numbers", input_schema={}, fn=lambda i: str(i["a"] + i["b"]))
    agent = Agent("a", model, tools=[tool])

    collected = list(agent.run_stream("what is 2+3"))

    tool_events = [e for e in collected if isinstance(e, ToolResultEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].name == "add"
    assert tool_events[0].result.content == "5"
    assert collected[-1].response.text == "the sum is 5"


async def test_arun_returns_final_response_using_agenerate():
    model = ScriptedModel("fake-1", [_final_text_response("async hello", "r1")])
    agent = Agent("a", model)

    response = await agent.arun("hi")

    assert response.text == "async hello"
    assert agent.memory.working.messages[-1].text == "async hello"


async def test_arun_stream_yields_events_using_astream():
    events = [[TextDelta(text="async "), MessageStop(response=_final_text_response("async hi", "r1"))]]
    model = ScriptedStreamModel("fake-1", events)
    agent = Agent("a", model)

    collected = [event async for event in agent.arun_stream("hi")]

    text = "".join(e.text for e in collected if isinstance(e, TextDelta))
    assert text == "async "
    assert collected[-1].response.text == "async hi"
