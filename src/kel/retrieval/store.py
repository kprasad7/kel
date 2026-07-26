"""VectorStore protocol (DESIGN.md 3.4): the interface any real backend
(Pinecone, Weaviate, Qdrant, Chroma, pgvector, Milvus) would implement to
plug into kel.retrieval.Retriever. v1 ships one concrete adapter —
InMemoryVectorStore — real vendor adapters are future work, same pattern
as kel.models' provider adapters (lazy-imported, registered by prefix)."""

from __future__ import annotations

import math
from typing import Any, Protocol

from kel.retrieval.types import Chunk, ScoredChunk

# Exact-match-on-every-key filtering (an implicit AND across fields) —
# the common case ("only this user's docs", "only Amazon listings") that
# every backend below supports natively in one form or another. Range
# queries, $in/$or, and other richer filter DSLs are a reasonable future
# upgrade behind the same `filter=` parameter, not something v1 needs to
# get right for the 80% case to be useful.
MetadataFilter = dict[str, Any]


def _matches_filter(metadata: dict[str, Any], filter: MetadataFilter | None) -> bool:
    if not filter:
        return True
    return all(metadata.get(key) == value for key, value in filter.items())


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def query(self, embedding: list[float], k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]: ...
    def keyword_query(self, query: str, k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def delete(self, ids: list[str]) -> None: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def delete(self, ids: list[str]) -> None:
        for chunk_id in ids:
            self._chunks.pop(chunk_id, None)

    def query(self, embedding: list[float], k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]:
        scored = [
            (_cosine_similarity(embedding, chunk.embedding), chunk)
            for chunk in self._chunks.values()
            if chunk.embedding is not None and _matches_filter(chunk.metadata, filter)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ScoredChunk(chunk=chunk, score=score) for score, chunk in scored[:k]]

    def keyword_query(self, query: str, k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]:
        """Naive term-frequency overlap — the "keyword" half of hybrid
        search. Not BM25; a real BM25/inverted-index implementation is a
        reasonable future upgrade behind this same method signature."""
        query_words = query.lower().split()
        scored: list[tuple[int, Chunk]] = []
        for chunk in self._chunks.values():
            if not _matches_filter(chunk.metadata, filter):
                continue
            words = chunk.text.lower().split()
            score = sum(words.count(w) for w in query_words)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ScoredChunk(chunk=chunk, score=float(score)) for score, chunk in scored[:k]]

    def __len__(self) -> int:
        return len(self._chunks)
