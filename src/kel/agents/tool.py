from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.models.types import ToolSpec


class Tool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any], fn: Callable[[dict], str]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn

    def to_spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self.input_schema)

    def __call__(self, tool_input: dict[str, Any]) -> str:
        return self.fn(tool_input)
