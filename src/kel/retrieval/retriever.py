"""Ingestion + retrieval pipeline over a VectorStore (DESIGN.md 3.4).
v1 implements naive top-k and hybrid (vector + keyword) search — graph-RAG
and agentic RAG are documented future stages, not implemented here; they'd
compose with this same Retriever rather than replace it (graph-RAG adds an
entity/relation traversal step, agentic RAG makes `retrieve` a tool the
agent calls iteratively via kel.agents)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from kel.retrieval.reranker import Reranker
from kel.retrieval.splitter import split_text
from kel.retrieval.store import MetadataFilter, VectorStore
from kel.retrieval.types import Chunk, ScoredChunk

Embedder = Callable[[str], list[float]]
Splitter = Callable[..., list[str]]

# how many first-stage candidates to overfetch before handing them to a
# reranker — a reranker needs a wider pool to actually improve on than
# the final k, but overfetching indefinitely just costs more without
# more benefit past a point.
_RERANK_POOL_SIZE = 20


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        splitter: Splitter = split_text,
        reranker: Reranker | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.splitter = splitter
        # Injected, not hardcoded — same DI shape as `store`/`embedder`/
        # `splitter` above. None (the default) means retrieval behaves
        # exactly as before; pass a `kel.retrieval.reranker.LLMReranker`
        # (or your own `Reranker`-shaped object) to add a second, more
        # expensive relevance pass over the first-stage candidates.
        self.reranker = reranker

    def ingest(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 500,
        overlap: int = 50,
        id_prefix: str | None = None,
    ) -> list[str]:
        prefix = id_prefix or uuid.uuid4().hex
        pieces = self.splitter(text, chunk_size=chunk_size, overlap=overlap)
        chunks = [
            Chunk(id=f"{prefix}-{i}", text=piece, metadata=metadata or {}, embedding=self.embedder(piece))
            for i, piece in enumerate(pieces)
        ]
        self.store.upsert(chunks)
        return [c.id for c in chunks]

    def retrieve(self, query: str, k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]:
        """Naive top-k vector similarity. `filter` scopes the search to
        chunks whose metadata matches every given key/value (e.g.
        `filter={"user_id": "u1"}`) — see `kel.retrieval.store.VectorStore`.
        If a `reranker` was given, overfetches a wider candidate pool and
        lets it re-score/re-order down to `k`."""
        pool_size = max(k, _RERANK_POOL_SIZE) if self.reranker is not None else k
        candidates = self.store.query(self.embedder(query), k=pool_size, filter=filter)
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, k=k)
        return candidates[:k]

    def retrieve_hybrid(
        self, query: str, k: int = 5, *, vector_weight: float = 0.5, filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Combine vector similarity and keyword overlap via a weighted
        sum after min-max normalizing each signal to [0, 1], so neither
        scale dominates just because it happens to run higher. If a
        `reranker` was given, overfetches a wider candidate pool and lets
        it re-score/re-order down to `k`."""
        pool_size = max(k, _RERANK_POOL_SIZE) if self.reranker is not None else k
        vector_hits = self.store.query(self.embedder(query), k=pool_size * 2, filter=filter)
        keyword_hits = self.store.keyword_query(query, k=pool_size * 2, filter=filter)

        vector_scores = {sc.chunk.id: sc.score for sc in vector_hits}
        keyword_scores = {sc.chunk.id: sc.score for sc in keyword_hits}
        max_keyword = max(keyword_scores.values(), default=0.0) or 1.0

        combined: dict[str, float] = {}
        for chunk_id in set(vector_scores) | set(keyword_scores):
            v = vector_scores.get(chunk_id, 0.0)
            kw = keyword_scores.get(chunk_id, 0.0) / max_keyword
            combined[chunk_id] = vector_weight * v + (1 - vector_weight) * kw

        ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)[:pool_size]
        results = []
        for chunk_id, score in ranked:
            chunk = self.store.get(chunk_id)
            if chunk is not None:
                results.append(ScoredChunk(chunk=chunk, score=score))

        if self.reranker is not None:
            return self.reranker.rerank(query, results, k=k)
        return results[:k]
