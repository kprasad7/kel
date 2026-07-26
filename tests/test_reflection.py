import pytest

from helpers import ScriptedModel
from kel.agents import Agent, areflect_and_retry, reflect_and_retry
from kel.models.types import ModelResponse, TextPart, Usage


def _response(text: str, rid: str = "r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())


def test_reflect_and_retry_accepts_on_first_attempt_when_critic_is_satisfied():
    model = ScriptedModel("fake-1", [_response("a good answer")])
    agent = Agent("a", model)

    result = reflect_and_retry(agent, "question", critic=lambda text: (True, ""))

    assert result.accepted is True
    assert result.attempts == 1
    assert result.response.text == "a good answer"


def test_reflect_and_retry_feeds_critique_back_and_retries():
    model = ScriptedModel(
        "fake-1", [_response("too short"), _response("a much more complete and thorough answer")]
    )
    agent = Agent("a", model)

    def critic(text):
        if len(text) < 20:
            return False, "the answer is too short, add more detail"
        return True, ""

    result = reflect_and_retry(agent, "question", critic=critic, max_attempts=3)

    assert result.accepted is True
    assert result.attempts == 2
    assert result.response.text == "a much more complete and thorough answer"
    # the second call's input should carry the critique backward to the generator
    second_call_messages = model.calls[1][0]
    combined = " ".join(m.text for m in second_call_messages)
    assert "too short, add more detail" in combined
    assert "question" in combined  # original request preserved too


def test_reflect_and_retry_gives_up_after_max_attempts_without_raising():
    model = ScriptedModel("fake-1", [_response("bad")] * 3)
    agent = Agent("a", model)

    result = reflect_and_retry(agent, "question", critic=lambda text: (False, "never good enough"), max_attempts=3)

    assert result.accepted is False
    assert result.attempts == 3
    assert result.response.text == "bad"  # last attempt's response, not an exception


def test_reflect_and_retry_rejects_max_attempts_below_one():
    model = ScriptedModel("fake-1", [])
    agent = Agent("a", model)

    with pytest.raises(ValueError):
        reflect_and_retry(agent, "question", critic=lambda text: (True, ""), max_attempts=0)


async def test_areflect_and_retry_feeds_critique_back_and_retries():
    model = ScriptedModel("fake-1", [_response("too short"), _response("a much better and longer answer")])
    agent = Agent("a", model)

    async def critic(text):
        if len(text) < 20:
            return False, "too short"
        return True, ""

    result = await areflect_and_retry(agent, "question", critic=critic, max_attempts=3)

    assert result.accepted is True
    assert result.attempts == 2
