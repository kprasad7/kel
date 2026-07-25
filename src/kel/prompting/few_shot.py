"""Few-shot prompting — the kel equivalent of LangChain's
`FewShotPromptTemplate`. Deliberately just string assembly, no new
templating engine: `str.format` is enough for this."""

from __future__ import annotations

from pydantic import BaseModel

_DEFAULT_EXAMPLE_TEMPLATE = "Input: {input}\nOutput: {output}"


class FewShotExample(BaseModel):
    input: str
    output: str


def format_few_shot_prompt(
    examples: list[FewShotExample],
    *,
    query: str,
    instructions: str | None = None,
    example_template: str = _DEFAULT_EXAMPLE_TEMPLATE,
) -> str:
    parts: list[str] = []
    if instructions:
        parts.append(instructions)
    parts.extend(example_template.format(input=ex.input, output=ex.output) for ex in examples)
    parts.append(f"Input: {query}\nOutput:")
    return "\n\n".join(parts)
