import tempfile
import time
from pathlib import Path

import pytest

from kel.specs import (
    AgentSpecLoader,
    load_agent_spec,
    load_eval_cases,
    parse_agent_spec,
    run_eval_suite,
)

AGENT_MD = """\
---
name: weather-agent
model: anthropic:claude-sonnet-5
api_key_ref: ANTHROPIC_API_KEY
tools:
  - get_weather
budget:
  max_tokens: 5000
version: "1.0"
owner: platform-team
---
You are a helpful weather assistant. Always cite the source of your data.
"""

EVAL_MD = """\
---
name: weather-agent-eval
cases:
  - name: basic
    input: "what's the weather in paris"
    expected_contains: ["Paris"]
  - name: fails-on-purpose
    input: "irrelevant"
    expected_contains: ["never gonna match"]
---
"""


def test_parse_agent_spec_reads_frontmatter_and_body():
    spec = parse_agent_spec(AGENT_MD)
    assert spec.name == "weather-agent"
    assert spec.model == "anthropic:claude-sonnet-5"
    assert spec.api_key_ref == "ANTHROPIC_API_KEY"
    assert spec.tools == ["get_weather"]
    assert spec.budget == {"max_tokens": 5000}
    assert spec.version == "1.0"
    assert spec.metadata == {"owner": "platform-team"}
    assert spec.system_prompt.startswith("You are a helpful weather assistant.")


def test_parse_agent_spec_requires_model():
    with pytest.raises(ValueError):
        parse_agent_spec("---\nname: no-model\n---\nbody")


def test_load_agent_spec_from_file_uses_filename_as_default_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weather.md"
        path.write_text("---\nmodel: openai:gpt-5.2\n---\nBe helpful.", encoding="utf-8")
        spec = load_agent_spec(path)
        assert spec.name == "weather"
        assert spec.model == "openai:gpt-5.2"


def test_agent_spec_loader_hot_reloads_on_mtime_change():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.md"
        path.write_text("---\nmodel: openai:gpt-5.2\n---\nversion one", encoding="utf-8")
        loader = AgentSpecLoader(path)
        first = loader.load()
        assert "version one" in first.system_prompt

        time.sleep(0.05)
        path.write_text("---\nmodel: openai:gpt-5.2\n---\nversion two", encoding="utf-8")
        second = loader.load()
        assert "version two" in second.system_prompt


def test_agent_spec_loader_caches_when_file_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.md"
        path.write_text("---\nmodel: openai:gpt-5.2\n---\nbody", encoding="utf-8")
        loader = AgentSpecLoader(path)
        first = loader.load()
        second = loader.load()
        assert first is second


def test_load_eval_cases_parses_cases_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval.md"
        path.write_text(EVAL_MD, encoding="utf-8")
        cases = load_eval_cases(path)
        assert [c.name for c in cases] == ["basic", "fails-on-purpose"]


def test_run_eval_suite_reports_pass_and_fail():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval.md"
        path.write_text(EVAL_MD, encoding="utf-8")
        cases = load_eval_cases(path)

        def fake_respond(prompt: str) -> str:
            return "The weather in Paris is sunny." if "paris" in prompt.lower() else "I don't know."

        results = run_eval_suite(cases, fake_respond)
        by_name = {r.case.name: r.passed for r in results}
        assert by_name["basic"] is True
        assert by_name["fails-on-purpose"] is False
