import pytest
from pydantic import BaseModel

from kel import Message, generate_structured, agenerate_structured
from kel.models.structured import StructuredOutputError
from kel.models.types import ModelResponse, TextPart, ToolUsePart, Usage
from helpers import ScriptedModel


class Person(BaseModel):
    name: str
    age: int


def _response(content, rid="r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=content, stop_reason="tool_use", usage=Usage())


def test_generate_structured_returns_validated_model_on_first_try():
    tool_call = ToolUsePart(id="1", name="return_structured_output", input={"name": "Ada", "age": 30})
    model = ScriptedModel("fake-1", [_response([tool_call])])

    person = generate_structured(model, [Message.user("who is Ada Lovelace")], Person)

    assert isinstance(person, Person)
    assert person.name == "Ada"
    assert person.age == 30


def test_generate_structured_retries_when_model_responds_with_plain_text():
    plain_text_response = _response([TextPart(text="Ada Lovelace was 30ish")], "r1")
    tool_call = ToolUsePart(id="1", name="return_structured_output", input={"name": "Ada", "age": 30})
    tool_response = _response([tool_call], "r2")
    model = ScriptedModel("fake-1", [plain_text_response, tool_response])

    person = generate_structured(model, [Message.user("who is Ada")], Person)

    assert person.name == "Ada"
    assert len(model.calls) == 2


def test_generate_structured_retries_on_validation_error():
    bad_call = ToolUsePart(id="1", name="return_structured_output", input={"name": "Ada", "age": "not a number"})
    good_call = ToolUsePart(id="2", name="return_structured_output", input={"name": "Ada", "age": 30})
    model = ScriptedModel("fake-1", [_response([bad_call], "r1"), _response([good_call], "r2")])

    person = generate_structured(model, [Message.user("who is Ada")], Person, max_retries=2)

    assert person.age == 30
    assert len(model.calls) == 2


def test_generate_structured_raises_after_exhausting_retries():
    bad_call = ToolUsePart(id="1", name="return_structured_output", input={"name": "Ada", "age": "bad"})
    model = ScriptedModel("fake-1", [_response([bad_call], f"r{i}") for i in range(5)])

    with pytest.raises(StructuredOutputError):
        generate_structured(model, [Message.user("who is Ada")], Person, max_retries=1)

    assert len(model.calls) == 2  # initial + 1 retry


async def test_agenerate_structured_returns_validated_model():
    tool_call = ToolUsePart(id="1", name="return_structured_output", input={"name": "Grace", "age": 40})
    model = ScriptedModel("fake-1", [_response([tool_call])])

    person = await agenerate_structured(model, [Message.user("who is Grace")], Person)

    assert person.name == "Grace"
