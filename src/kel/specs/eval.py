"""Minimal eval-suite runner over a `.md` spec's `cases:` frontmatter
(DESIGN.md 3.6: "kel eval command runs an agent spec against its eval
cases and diffs against the last known-good run"). v1 implements the
substring-assertion case here; the diff-against-last-known-good part
belongs to kel.testing's golden-trace comparison, not duplicated here."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kel.specs.frontmatter import split_frontmatter
from kel.specs.types import EvalCase


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    text = Path(path).read_text(encoding="utf-8")
    data, _ = split_frontmatter(text)
    return [EvalCase(**case) for case in data.get("cases", [])]


class EvalResult:
    def __init__(self, case: EvalCase, output: str, passed: bool):
        self.case = case
        self.output = output
        self.passed = passed


def run_eval_case(case: EvalCase, respond: Callable[[str], str]) -> EvalResult:
    output = respond(case.input)
    passed = all(fragment in output for fragment in case.expected_contains)
    return EvalResult(case, output, passed)


def run_eval_suite(cases: list[EvalCase], respond: Callable[[str], str]) -> list[EvalResult]:
    return [run_eval_case(case, respond) for case in cases]
