"""LLM-graded evaluation: use a judge model to score correctness against
free-form criteria. Built directly on `generate_structured` (no new
mechanism): the judge model is asked to return a validated `Grade` object,
not free text you'd have to parse yourself.

Falls back to substring matching (`kel.specs.eval.run_eval_case`'s
behavior) for any `EvalCase` that doesn't set `criteria` — a spec file can
mix both grading styles across its cases.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from kel.models.base import ChatModel
from kel.models.structured import generate_structured
from kel.models.types import Message
from kel.specs.eval import EvalResult
from kel.specs.types import EvalCase


class Grade(BaseModel):
    passed: bool
    score: float
    reasoning: str


def llm_judge(
    model: ChatModel, *, input: str, output: str, criteria: str, reference: str | None = None
) -> Grade:
    reference_line = f"\nReference/expected answer (for comparison, not required to match exactly): {reference}" if reference else ""
    prompt = (
        "You are grading an AI assistant's response.\n\n"
        f"Input given to the assistant: {input}\n"
        f"Assistant's response: {output}\n"
        f"Criteria: {criteria}"
        f"{reference_line}\n\n"
        "Grade whether the response satisfies the criteria. Give a score from "
        "0.0 (completely fails) to 1.0 (fully satisfies), and set passed=true "
        "only if score >= 0.7."
    )
    return generate_structured(model, [Message.user(prompt)], Grade)


def run_llm_graded_eval_case(
    model: ChatModel, case: EvalCase, respond: Callable[[str], str]
) -> EvalResult:
    output = respond(case.input)
    if case.criteria:
        grade = llm_judge(model, input=case.input, output=output, criteria=case.criteria)
        result = EvalResult(case, output, grade.passed)
        result.grade = grade
        return result
    passed = all(fragment in output for fragment in case.expected_contains)
    return EvalResult(case, output, passed)


def run_llm_graded_eval_suite(
    model: ChatModel, cases: list[EvalCase], respond: Callable[[str], str]
) -> list[EvalResult]:
    return [run_llm_graded_eval_case(model, case, respond) for case in cases]
