"""Qdrant adapter for `kel.retrieval.VectorStore`. Requires the
`qdrant-client` package (`pip install kel[qdrant]`).

Same caveat as the Cohere adapter, stated plainly: written against
Qdrant's documented client API (`query_points`/`upsert`/`retrieve`/
`scroll`), not exercised against a live Qdrant instance — tested here
against an injected fake client that mimics that documented shape.

Uses `query_points` (returns a `QueryResponse` with a `.points` list),
not the older `search` method — `search` was removed in qdrant-client
1.10+, and pinning below that just to keep `search` working is not a
real fix.

One real constraint worth calling out: **Qdrant point IDs must be an
unsigned integer or a UUID** — kel's `Chunk.id` is an arbitrary string
(e.g. `"doc1-0"`), which Qdrant would reject outright. This adapter maps
each chunk id to a deterministic UUID (`uuid5` of the chunk id) for the
Qdrant point id, and stores the original id in the payload (`chunk_id`)
so results map back to the right `Chunk.id` — this translation is the
actual reason this file exists rather than being a thin pass-through.
"""

from __future__ import annotations

import uuid
from typing import Any

from kel.retrieval.store import MetadataFilter
from kel.retrieval.types import Chunk, ScoredChunk

_UUID_NAMESPACE = uuid.UUID("6f6e2b1a-2f2a-4b8a-9b0a-1a2b3c4d5e6f")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


def _build_filter(filter: MetadataFilter | None, extra_conditions: list[Any] | None = None) -> Any:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = list(extra_conditions or [])
    for key, value in (filter or {}).items():
        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        vector_size: int | None = None,
        client: Any = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._collection_ready = False
        if client is not None:
            self._client = client
        else:
            try:
                import qdrant_client
            except ImportError as exc:
                raise ImportError(
                    "The qdrant-client package is required to use QdrantVectorStore. "
                    "Install it with `pip install kel[qdrant]`."
                ) from exc
            self._client = qdrant_client.QdrantClient(url=url, api_key=api_key)

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            self._client.create_payload_index(
                collection_name=self.collection_name, field_name="text", field_schema=PayloadSchemaType.TEXT
            )
        self._collection_ready = True

    def upsert(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct

        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        first_embedding = embedded[0].embedding
        assert first_embedding is not None
        self._ensure_collection(self.vector_size or len(first_embedding))
        points = [
            PointStruct(
                id=_point_id(chunk.id),
                vector=chunk.embedding,
                payload={"chunk_id": chunk.id, "text": chunk.text, **chunk.metadata},
            )
            for chunk in embedded
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(self, embedding: list[float], k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]:
        if not self._collection_ready:
            return []
        response = self._client.query_points(
            collection_name=self.collection_name, query=embedding, limit=k, query_filter=_build_filter(filter)
        )
        return [self._to_scored_chunk(hit.payload, hit.score) for hit in response.points]

    def keyword_query(self, query: str, k: int = 5, *, filter: MetadataFilter | None = None) -> list[ScoredChunk]:
        if not self._collection_ready:
            return []
        from qdrant_client.models import FieldCondition, MatchText

        text_condition = FieldCondition(key="text", match=MatchText(text=query))
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=_build_filter(filter, extra_conditions=[text_condition]),
            limit=k,
        )
        return [self._to_scored_chunk(point.payload, 1.0) for point in points]

    def get(self, chunk_id: str) -> Chunk | None:
        if not self._collection_ready:
            return None
        records = self._client.retrieve(collection_name=self.collection_name, ids=[_point_id(chunk_id)])
        if not records:
            return None
        return self._to_chunk(records[0].payload)

    def delete(self, ids: list[str]) -> None:
        if not self._collection_ready or not ids:
            return
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self.collection_name, points_selector=PointIdsList(points=[_point_id(i) for i in ids])
        )

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> Chunk:
        metadata = {k: v for k, v in payload.items() if k not in ("chunk_id", "text")}
        return Chunk(id=payload["chunk_id"], text=payload.get("text", ""), metadata=metadata)

    @classmethod
    def _to_scored_chunk(cls, payload: dict[str, Any], score: float) -> ScoredChunk:
        return ScoredChunk(chunk=cls._to_chunk(payload), score=score)
