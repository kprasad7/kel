from helpers import ScriptedModel
from kel.agents.hallucination import HallucinationChecker
from kel.models.types import ModelResponse, ToolUsePart, Usage


def _structured_response(grounded: bool, unsupported_claims: list[str], confidence: float, rid: str = "r") -> ModelResponse:
    tool_call = ToolUsePart(
        id="1",
        name="return_structured_output",
        input={"grounded": grounded, "unsupported_claims": unsupported_claims, "confidence": confidence},
    )
    return ModelResponse(id=rid, model="fake-1", content=[tool_call], stop_reason="tool_use", usage=Usage())


def test_check_reports_grounded_when_model_finds_no_unsupported_claims():
    model = ScriptedModel("fake-1", [_structured_response(True, [], 0.95)])
    checker = HallucinationChecker(model)

    report = checker.check("Paris is the capital of France.", sources="France's capital is Paris.")

    assert report.grounded is True
    assert report.unsupported_claims == []
    assert report.confidence == 0.95


def test_check_reports_unsupported_claims_when_response_fabricates_facts():
    model = ScriptedModel(
        "fake-1", [_structured_response(False, ["the Eiffel Tower is 500m tall"], 0.8)]
    )
    checker = HallucinationChecker(model)

    report = checker.check(
        "Paris is the capital of France. The Eiffel Tower is 500m tall.",
        sources="France's capital is Paris.",
    )

    assert report.grounded is False
    assert report.unsupported_claims == ["the Eiffel Tower is 500m tall"]


def test_check_accepts_a_list_of_sources_and_joins_them():
    model = ScriptedModel("fake-1", [_structured_response(True, [], 0.9)])
    checker = HallucinationChecker(model)

    report = checker.check("summary text", sources=["source chunk one", "source chunk two"])

    assert report.grounded is True
    prompt = model.calls[0][0][-1].text
    assert "source chunk one" in prompt
    assert "source chunk two" in prompt


async def test_acheck_reports_grounded_when_model_finds_no_unsupported_claims():
    model = ScriptedModel("fake-1", [_structured_response(True, [], 0.9)])
    checker = HallucinationChecker(model)

    report = await checker.acheck("Paris is the capital of France.", sources="France's capital is Paris.")

    assert report.grounded is True
