from kel.specs import EvalCase, Grade, llm_judge, run_llm_graded_eval_case, run_llm_graded_eval_suite
from kel.models.types import ModelResponse, ToolUsePart, Usage
from helpers import ScriptedModel


def _grade_response(passed: bool, score: float, reasoning: str, rid="r") -> ModelResponse:
    tool_call = ToolUsePart(
        id="1", name="return_structured_output", input={"passed": passed, "score": score, "reasoning": reasoning}
    )
    return ModelResponse(id=rid, model="fake-1", content=[tool_call], stop_reason="tool_use", usage=Usage())


def test_llm_judge_returns_validated_grade():
    model = ScriptedModel("fake-1", [_grade_response(True, 0.9, "clearly satisfies the criteria")])

    grade = llm_judge(model, input="what is 2+2", output="4", criteria="the answer is mathematically correct")

    assert isinstance(grade, Grade)
    assert grade.passed is True
    assert grade.score == 0.9


def test_run_llm_graded_eval_case_uses_judge_when_criteria_set():
    model = ScriptedModel("fake-1", [_grade_response(False, 0.2, "response is incorrect")])
    case = EvalCase(name="math-check", input="what is 2+2", criteria="the answer is mathematically correct")

    result = run_llm_graded_eval_case(model, case, respond=lambda text: "5")

    assert result.passed is False
    assert result.grade.reasoning == "response is incorrect"


def test_run_llm_graded_eval_case_falls_back_to_substring_match_without_criteria():
    model = ScriptedModel("fake-1", [])  # never called — no criteria means no judge call
    case = EvalCase(name="basic", input="say hi", expected_contains=["hello"])

    result = run_llm_graded_eval_case(model, case, respond=lambda text: "hello there")

    assert result.passed is True
    assert model.calls == []


def test_run_llm_graded_eval_suite_runs_all_cases():
    model = ScriptedModel(
        "fake-1", [_grade_response(True, 1.0, "good"), _grade_response(True, 0.8, "also good")]
    )
    cases = [
        EvalCase(name="c1", input="q1", criteria="answers the question"),
        EvalCase(name="c2", input="q2", criteria="answers the question"),
    ]

    results = run_llm_graded_eval_suite(model, cases, respond=lambda text: "an answer")

    assert [r.passed for r in results] == [True, True]
