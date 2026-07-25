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

from kel.retrieval.splitter import split_text
from kel.retrieval.store import VectorStore
from kel.retrieval.types import Chunk, ScoredChunk

Embedder = Callable[[str], list[float]]
Splitter = Callable[..., list[str]]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, *, splitter: Splitter = split_text):
        self.store = store
        self.embedder = embedder
        self.splitter = splitter

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

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Naive top-k vector similarity."""
        return self.store.query(self.embedder(query), k=k)

    def retrieve_hybrid(self, query: str, k: int = 5, *, vector_weight: float = 0.5) -> list[ScoredChunk]:
        """Combine vector similarity and keyword overlap via a weighted
        sum after min-max normalizing each signal to [0, 1], so neither
        scale dominates just because it happens to run higher."""
        vector_hits = self.store.query(self.embedder(query), k=k * 2)
        keyword_hits = self.store.keyword_query(query, k=k * 2)

        vector_scores = {sc.chunk.id: sc.score for sc in vector_hits}
        keyword_scores = {sc.chunk.id: sc.score for sc in keyword_hits}
        max_keyword = max(keyword_scores.values(), default=0.0) or 1.0

        combined: dict[str, float] = {}
        for chunk_id in set(vector_scores) | set(keyword_scores):
            v = vector_scores.get(chunk_id, 0.0)
            kw = keyword_scores.get(chunk_id, 0.0) / max_keyword
            combined[chunk_id] = vector_weight * v + (1 - vector_weight) * kw

        ranked = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)[:k]
        results = []
        for chunk_id, score in ranked:
            chunk = self.store.get(chunk_id)
            if chunk is not None:
                results.append(ScoredChunk(chunk=chunk, score=score))
        return results
