"""kel CLI (DESIGN.md 3.13): `kel run`, `kel eval`, `kel trace`. Not
implemented: `kel init <template>` project scaffolding — genuinely useful,
but picking/maintaining template content is separate work this pass
didn't do; these three commands are real and tested, not placeholders."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from kel.agents.agent import Agent
from kel.sdk.build import build_agent_from_spec
from kel.specs.eval import load_eval_cases, run_eval_suite
from kel.testing.cassette import Cassette

AgentBuilder = Callable[..., Agent]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kel")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an agent spec against a single input")
    run_p.add_argument("spec", help="path to an agent .md spec")
    run_p.add_argument("input", help="user input text")

    eval_p = sub.add_parser("eval", help="Run an agent spec against an eval-case .md file")
    eval_p.add_argument("spec", help="path to an agent .md spec")
    eval_p.add_argument("eval_file", help="path to an eval-case .md file")

    trace_p = sub.add_parser("trace", help="Inspect a recorded cassette")
    trace_p.add_argument("cassette", help="path to a cassette JSON file")

    return parser


def main(argv: list[str] | None = None, *, agent_builder: AgentBuilder = build_agent_from_spec) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        agent = agent_builder(args.spec)
        response = agent.run(args.input)
        print(response.text)
        return 0

    if args.command == "eval":
        agent = agent_builder(args.spec)
        cases = load_eval_cases(args.eval_file)
        results = run_eval_suite(cases, lambda text: agent.run(text).text)
        for result in results:
            print(f"[{'PASS' if result.passed else 'FAIL'}] {result.case.name}")
        passed = sum(1 for r in results if r.passed)
        print(f"{passed}/{len(results)} passed")
        return 0 if passed == len(results) else 1

    if args.command == "trace":
        cassette = Cassette.load(args.cassette)
        for i, interaction in enumerate(cassette.interactions):
            response = interaction.response
            preview = response.text[:80]
            print(
                f"[{i}] stop_reason={response.stop_reason} "
                f"tokens_in={response.usage.input_tokens} tokens_out={response.usage.output_tokens} "
                f"text={preview!r}"
            )
        return 0

    return 1


def run_cli() -> None:
    sys.exit(main())
