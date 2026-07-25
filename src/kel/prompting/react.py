"""Classic text-based ReAct prompting (Thought/Action/Observation loop).

Note this is *not* how `kel.agents.Agent` actually calls tools — that uses
native provider tool-calling (structured, not text-parsed), which is more
reliable and is what you should use for real agents. This module exists
for the cases native tool-calling doesn't cover: models/APIs without tool
support, teaching/demoing the ReAct pattern, or reproducing classic ReAct
papers/benchmarks that assume text-based Thought/Action/Observation."""

from __future__ import annotations

from kel.agents.tool import Tool

REACT_SYSTEM_TEMPLATE = """Answer the following question as best you can. You have access to these tools:

{tool_descriptions}

Use this format:
Thought: reason about what to do next
Action: the tool to use
Action Input: the input to the tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question"""


def build_react_system_prompt(tools: list[Tool]) -> str:
    descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return REACT_SYSTEM_TEMPLATE.format(tool_descriptions=descriptions)
