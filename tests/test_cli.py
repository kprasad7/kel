import tempfile
from pathlib import Path

import pytest

from helpers import ScriptedModel
from kel import __version__
from kel.agents.agent import Agent
from kel.models.types import ModelResponse, TextPart, Usage
from kel.sdk.cli import main
from kel.testing.cassette import Cassette, Interaction


def _fake_agent_builder_factory(response_text: str):
    def builder(spec_path: str, **kwargs) -> Agent:
        model = ScriptedModel(
            "fake-1",
            [ModelResponse(id="r", model="fake-1", content=[TextPart(text=response_text)], stop_reason="end_turn", usage=Usage())],
        )
        return Agent("cli-agent", model)

    return builder


def test_bare_invocation_shows_banner_and_help(capsys):
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: kel" in captured.out
    assert __version__ in captured.out


def test_version_flag_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert __version__ in captured.out


def test_run_command_prints_agent_response(capsys):
    exit_code = main(["run", "agent.md", "hello"], agent_builder=_fake_agent_builder_factory("hi there"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "hi there" in captured.out


def test_eval_command_reports_pass_and_exit_code(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "eval.md"
        eval_path.write_text(
            "---\ncases:\n  - name: greet\n    input: hi\n    expected_contains: [\"hi there\"]\n---\n",
            encoding="utf-8",
        )
        exit_code = main(
            ["eval", "agent.md", str(eval_path)], agent_builder=_fake_agent_builder_factory("hi there, friend")
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "[PASS] greet" in captured.out
        assert "1/1 passed" in captured.out


def test_eval_command_returns_nonzero_exit_code_on_failure(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "eval.md"
        eval_path.write_text(
            "---\ncases:\n  - name: greet\n    input: hi\n    expected_contains: [\"never matches\"]\n---\n",
            encoding="utf-8",
        )
        exit_code = main(
            ["eval", "agent.md", str(eval_path)], agent_builder=_fake_agent_builder_factory("hi there")
        )
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[FAIL] greet" in captured.out


def test_trace_command_prints_cassette_interactions(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        cassette_path = Path(tmp) / "cassette.json"
        cassette = Cassette(
            interactions=[
                Interaction(
                    request={"messages": []},
                    response=ModelResponse(
                        id="r1", model="fake-1", content=[TextPart(text="hello")], stop_reason="end_turn",
                        usage=Usage(input_tokens=3, output_tokens=2),
                    ),
                )
            ]
        )
        cassette.save(cassette_path)

        exit_code = main(["trace", str(cassette_path)])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "tokens_in=3" in captured.out
        assert "tokens_out=2" in captured.out
        assert "hello" in captured.out
