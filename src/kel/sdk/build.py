"""The "don't configure all the subsystems yourself" convenience layer
(DESIGN.md 3.13): turn a `.md` agent spec straight into a runnable Agent,
resolving its model/credentials/budget from the spec."""

from __future__ import annotations

import os

from kel.agents.agent import Agent
from kel.agents.tool import Tool
from kel.budget.types import Budget
from kel.models.registry import get_model
from kel.specs.loader import load_agent_spec


def build_agent_from_spec(
    path: str, *, tools: list[Tool] | None = None, api_key: str | None = None, instrument: bool = True
) -> Agent:
    spec = load_agent_spec(path)

    resolved_key = api_key
    if resolved_key is None and spec.api_key_ref:
        resolved_key = os.environ.get(spec.api_key_ref)

    model_kwargs: dict = {"instrument": instrument}
    if resolved_key is not None:
        model_kwargs["api_key"] = resolved_key
    if spec.budget:
        model_kwargs["budget"] = Budget(**spec.budget)

    model = get_model(spec.model, **model_kwargs)
    return Agent(spec.name, model, system_prompt=spec.system_prompt, tools=tools or [])
