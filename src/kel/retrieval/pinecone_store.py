"""Pinecone adapter for `kel.retrieval.VectorStore`. Requires
`pip install kel[pinecone]`.

Same documented-shape caveat as the other real-vendor adapters: written
against Pinecone's documented client API, tested against an injected fake
index, not exercised against a live Pinecone project.

Pinecone is a SaaS-only service with no ambient/IAM-role credential path
(unlike S3 or RDS) — it always needs an API key. Still read from
`PINECONE_API_KEY` when not given explicitly, so the key can come from a
Kubernetes secret mounted as an env var rather than being hardcoded.

Known limitation: Pinecone's base vector index has no built-in full-text/
keyword search (that needs a separate sparse index and hybrid setup this
adapter doesn't build) — `keyword_query` returns an empty list.
"""

from __future__ import annotations

import os
from typing import Any

from kel.retrieval.types import Chunk, ScoredChunk


class PineconeVectorStore:
    def __init__(
        self,
        index_name: str,
        *,
        api_key: str | None = None,
        dimension: int | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        client: Any = None,
        index: Any = None,
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self._index = index
        if index is not None or client is not None:
            self._client = client
        else:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise ImportError(
                    "The pinecone package is required to use PineconeVectorStore. "
                    "Install it with `pip install kel[pinecone]`."
                ) from exc
            self._client = Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY"))

    def _ensure_index(self, dimension: int) -> Any:
        if self._index is not None:
            return self._index
        from pinecone import ServerlessSpec

        existing = [i["name"] for i in self._client.list_indexes()]
        if self.index_name not in existing:
            self._client.create_index(
                name=self.index_name,
                dimension=self.dimension or dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = self._client.Index(self.index_name)
        return self._index

    def upsert(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        index = self._ensure_index(len(embedded[0].embedding))
        vectors = [
            {"id": c.id, "values": c.embedding, "metadata": {"text": c.text, **c.metadata}} for c in embedded
        ]
        index.upsert(vectors=vectors)

    def query(self, embedding: list[float], k: int = 5) -> list[ScoredChunk]:
        if self._index is None:
            return []
        result = self._index.query(vector=embedding, top_k=k, include_metadata=True)
        return [ScoredChunk(chunk=self._to_chunk(m.id, m.metadata), score=m.score) for m in result.matches]

    def keyword_query(self, query: str, k: int = 5) -> list[ScoredChunk]:
        return []

    def get(self, chunk_id: str) -> Chunk | None:
        if self._index is None:
            return None
        result = self._index.fetch(ids=[chunk_id])
        vec = result.vectors.get(chunk_id)
        if vec is None:
            return None
        return self._to_chunk(chunk_id, vec.metadata)

    def delete(self, ids: list[str]) -> None:
        if self._index is None or not ids:
            return
        self._index.delete(ids=ids)

    @staticmethod
    def _to_chunk(chunk_id: str, metadata: dict[str, Any] | None) -> Chunk:
        metadata = dict(metadata or {})
        text = metadata.pop("text", "")
        return Chunk(id=chunk_id, text=text, metadata=metadata)
