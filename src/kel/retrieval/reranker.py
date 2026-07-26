"""LLM-based reranking: a second, more expensive relevance pass over a
first-stage retriever's candidates. Hybrid (vector + keyword) search is a
fixed weighted-sum heuristic — a real quality lever most serious RAG
stacks add on top is a reranker that actually reads each candidate
against the query. Implemented as one structured-output call over every
candidate at once (not one call per candidate), so reranking k candidates
costs one model call, not k of them, and built on `generate_structured`
kel already has rather than a new mechanism or a required new dependency
(a cross-encoder model would need one; an LLM-based reranker needs
nothing beyond a `ChatModel` you already have).

Injectable, not hardcoded: `Retriever(store, embedder, reranker=...)` —
same dependency-injection shape as every other pluggable piece of kel
(the model gateway's `client=`, `Agent`'s `approve_tool_call`, etc.).
Retrieval works exactly as before when no reranker is given.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from kel.models.base import ChatModel
from kel.models.structured import agenerate_structured, generate_structured
from kel.models.types import Message
from kel.retrieval.types import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class _RelevanceScores(BaseModel):
    scores: list[float] = Field(
        description="One relevance score per candidate, in the same order as given, from 0.0 (irrelevant) to 1.0 (highly relevant)."
    )


def _build_prompt(query: str, candidates: list[ScoredChunk]) -> str:
    listed = "\n\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(candidates))
    return (
        f"Query: {query}\n\n"
        f"Candidate passages:\n{listed}\n\n"
        f"Score how relevant each candidate is to the query, from 0.0 (irrelevant) to 1.0 "
        f"(highly relevant). Return exactly {len(candidates)} scores, in the same order as "
        f"the candidates above."
    )


class LLMReranker:
    def __init__(self, model: ChatModel):
        self.model = model

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        result = generate_structured(self.model, [Message.user(_build_prompt(query, candidates))], _RelevanceScores)
        if len(result.scores) != len(candidates):
            # the model didn't follow the "one score per candidate" contract
            # — rather than risk silently mismatching a score to the wrong
            # chunk, fall back to the first-stage order/scores unchanged.
            return candidates[:k]
        rescored = [
            ScoredChunk(chunk=c.chunk, score=score) for c, score in zip(candidates, result.scores, strict=True)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]

    async def arerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        result = await agenerate_structured(
            self.model, [Message.user(_build_prompt(query, candidates))], _RelevanceScores
        )
        if len(result.scores) != len(candidates):
            return candidates[:k]
        rescored = [
            ScoredChunk(chunk=c.chunk, score=score) for c, score in zip(candidates, result.scores, strict=True)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]
