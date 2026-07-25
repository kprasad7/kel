"""Long-term facts. v1 default is dependency-free keyword-overlap search so
SemanticMemory is genuinely usable with zero setup; pass an `embedder`
callable (e.g. backed by kel.retrieval's embedding model once that's wired
up) to upgrade to real vector similarity without changing the API."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class SemanticFact(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMemory:
    def __init__(self, embedder: Callable[[str], list[float]] | None = None):
        self._facts: dict[str, SemanticFact] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedder = embedder

    def remember(self, text: str, *, metadata: dict[str, Any] | None = None, id: str | None = None) -> str:
        fact_id = id or uuid.uuid4().hex
        self._facts[fact_id] = SemanticFact(id=fact_id, text=text, metadata=metadata or {})
        if self._embedder is not None:
            self._embeddings[fact_id] = self._embedder(text)
        return fact_id

    def forget(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)
        self._embeddings.pop(fact_id, None)

    def search(self, query: str, k: int = 5) -> list[SemanticFact]:
        if not self._facts:
            return []
        if self._embedder is not None:
            query_vec = self._embedder(query)
            scored = [
                (_cosine_similarity(query_vec, self._embeddings[fid]), fact)
                for fid, fact in self._facts.items()
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [fact for _, fact in scored[:k]]

        query_words = set(query.lower().split())

        def overlap(fact: SemanticFact) -> int:
            return len(query_words & set(fact.text.lower().split()))

        ranked = sorted(self._facts.values(), key=overlap, reverse=True)
        matches = [f for f in ranked if overlap(f) > 0]
        return matches[:k] if matches else ranked[:k]

    def __len__(self) -> int:
        return len(self._facts)
