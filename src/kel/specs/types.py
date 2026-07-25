from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """An agent defined as a `.md` file with YAML frontmatter (DESIGN.md
    3.6 + 3.11's "declared in the agent's .md spec frontmatter" credential
    model). The frontmatter carries structured config; the markdown body
    below it is the system prompt."""

    name: str
    model: str
    """"provider:model_id", e.g. "anthropic:claude-sonnet-5"."""
    system_prompt: str
    api_key_ref: str | None = None
    """Env var name to read the API key from — credential choice is
    versioned with the spec, not buried in ambient environment setup."""
    tools: list[str] = Field(default_factory=list)
    budget: dict[str, Any] | None = None
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    name: str
    input: str
    expected_contains: list[str] = Field(default_factory=list)
    criteria: str | None = None
    """Free-text grading criteria (e.g. "the response is helpful and cites
    a source") — when set, `run_llm_graded_eval_case` uses an LLM judge
    instead of substring matching. See `kel.specs.llm_eval`."""
    metadata: dict[str, Any] = Field(default_factory=dict)
