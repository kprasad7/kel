"""Long-term facts. v1 default is dependency-free keyword-overlap search so
SemanticMemory is genuinely usable with zero setup; pass an `embedder`
callable (e.g. backed by kel.retrieval's embedding model once that's wired
up) to upgrade to real vector similarity without changing the API.

**Expiration is opt-in per fact, not a forced two-tier taxonomy.**
`remember(..., ttl_seconds=...)` marks a fact as decaying; leave it unset
(the default) and the fact never expires — the same "foundational fact
vs. passing chatter" distinction some frameworks bake into a rigid
multi-tier memory architecture, done here as one optional parameter
instead of a new abstraction. `search()` never returns an expired fact;
`forget_expired()` purges them from storage."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class SemanticFact(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float | None = None
    """None (the default) means the fact never expires. Set to decay it
    after this many seconds from `created_at` — e.g. a passing remark
    worth remembering for a session but not indefinitely."""

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now if now is not None else time.time()) - self.created_at > self.ttl_seconds


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
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

    def remember(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        fact_id = id or uuid.uuid4().hex
        self._facts[fact_id] = SemanticFact(id=fact_id, text=text, metadata=metadata or {}, ttl_seconds=ttl_seconds)
        if self._embedder is not None:
            self._embeddings[fact_id] = self._embedder(text)
        return fact_id

    def forget(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)
        self._embeddings.pop(fact_id, None)

    def forget_expired(self, *, now: float | None = None) -> int:
        """Purges every expired fact from storage. Not required for
        `search()` to stay correct (it already filters expired facts
        itself) — this is for reclaiming storage/keeping `len()`
        accurate over a long-lived process."""
        expired = [fid for fid, fact in self._facts.items() if fact.is_expired(now=now)]
        for fid in expired:
            self.forget(fid)
        return len(expired)

    def search(self, query: str, k: int = 5) -> list[SemanticFact]:
        live_facts = {fid: fact for fid, fact in self._facts.items() if not fact.is_expired()}
        if not live_facts:
            return []
        if self._embedder is not None:
            query_vec = self._embedder(query)
            scored = [
                (_cosine_similarity(query_vec, self._embeddings[fid]), fact) for fid, fact in live_facts.items()
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [fact for _, fact in scored[:k]]

        query_words = set(query.lower().split())

        def overlap(fact: SemanticFact) -> int:
            return len(query_words & set(fact.text.lower().split()))

        ranked = sorted(live_facts.values(), key=overlap, reverse=True)
        matches = [f for f in ranked if overlap(f) > 0]
        return matches[:k] if matches else ranked[:k]

    def __len__(self) -> int:
        return len(self._facts)
