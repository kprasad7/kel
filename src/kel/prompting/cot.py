"""Chain-of-thought prompting: ask the model to reason before answering,
then reliably extract just the final answer out of the reasoning trace."""

from __future__ import annotations

DEFAULT_COT_SUFFIX = (
    "Think step by step before giving your final answer. Show your reasoning, "
    "then clearly state the final answer prefixed with 'Final Answer:'."
)


def chain_of_thought_prompt(question: str, *, suffix: str = DEFAULT_COT_SUFFIX) -> str:
    return f"{question}\n\n{suffix}"


def extract_final_answer(text: str, *, marker: str = "Final Answer:") -> str:
    """Pulls the text after the last occurrence of `marker`. Falls back to
    the whole (stripped) text if the model didn't follow the requested
    format — a missing marker is a real, expected outcome, not exceptional."""
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker) :].strip()
