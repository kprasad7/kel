"""Hallucination validation: a second pass that checks whether a
response's claims are actually supported by given source material —
retrieved RAG chunks, tool results, or any other context the response
was supposed to be grounded in. Same one-call structured-output pattern
`kel.retrieval.reranker.LLMReranker` uses, not a new mechanism, and no
new required dependency (works with any `ChatModel` already in hand).

Not wired into `Agent` automatically — same "composable, not forced"
shape as `LLMReranker` not being wired into every `Retriever` call. Run
it yourself after `agent.run(...)`, or compose it into a graph as its own
node via `kel.agents.agent_node()`:

```python
checker = HallucinationChecker(model)
response = agent.run("summarize the incident report")
report = checker.check(response.text, sources=retrieved_chunks_text)
if not report.grounded:
    ...  # retry, ask for citations, escalate to a human, etc.
```

Not a guarantee — it's still one LLM judging another LLM's output, not a
formal proof of correctness — but a real, concrete second check where
today there is none.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kel.models.base import ChatModel
from kel.models.structured import agenerate_structured, generate_structured
from kel.models.types import Message


class HallucinationReport(BaseModel):
    grounded: bool = Field(description="True only if every factual claim in the response is supported by the sources.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims from the response that are fabricated, unsupported, or contradicted by the sources. Empty if grounded is true.",
    )
    confidence: float = Field(description="0.0 (not confident) to 1.0 (fully confident) in this judgment.")


def _build_prompt(response_text: str, sources: str) -> str:
    return (
        f"Sources:\n{sources}\n\n"
        f"Response to check:\n{response_text}\n\n"
        "Determine whether every factual claim in the response is actually supported by "
        "the sources above. A claim is unsupported if it is fabricated, not present in the "
        "sources, or contradicts them — restating information the sources genuinely contain "
        "is fine. List every unsupported claim verbatim in `unsupported_claims`. Set "
        "`grounded=true` only if that list would be empty."
    )


class HallucinationChecker:
    def __init__(self, model: ChatModel):
        self.model = model

    def check(self, response_text: str, sources: str | list[str]) -> HallucinationReport:
        joined = sources if isinstance(sources, str) else "\n\n".join(sources)
        return generate_structured(self.model, [Message.user(_build_prompt(response_text, joined))], HallucinationReport)

    async def acheck(self, response_text: str, sources: str | list[str]) -> HallucinationReport:
        joined = sources if isinstance(sources, str) else "\n\n".join(sources)
        return await agenerate_structured(
            self.model, [Message.user(_build_prompt(response_text, joined))], HallucinationReport
        )
