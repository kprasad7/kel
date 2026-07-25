"""Weaviate adapter for `kel.retrieval.VectorStore`. Requires
`pip install kel[weaviate]`.

Same documented-shape caveat as the other real-vendor adapters.

**No API key required for self-hosted Weaviate** — a very common pattern
when Weaviate runs as an in-cluster service (e.g. a Kubernetes StatefulSet
alongside your app) with anonymous access enabled, which is the default
when `url` isn't given here (`weaviate.connect_to_local()`). An `api_key`
is only relevant for Weaviate Cloud; pass a `url` to switch to that path.

Same UUID constraint as Qdrant: Weaviate object IDs must be valid UUIDs,
so `Chunk.id` (an arbitrary string) is mapped to a deterministic UUID via
`uuid5`, with the original id stored in the object's properties.
"""

from __future__ import annotations

import uuid
from typing import Any

from kel.retrieval.types import Chunk, ScoredChunk

_UUID_NAMESPACE = uuid.UUID("7a8b9c0d-1e2f-4a3b-8c4d-5e6f7a8b9c0d")


def _object_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


class WeaviateVectorStore:
    def __init__(
        self, collection_name: str, *, url: str | None = None, api_key: str | None = None, client: Any = None
    ):
        self.collection_name = collection_name
        self._collection_ready = False
        if client is not None:
            self._client = client
        else:
            try:
                import weaviate
                from weaviate.auth import AuthApiKey
            except ImportError as exc:
                raise ImportError(
                    "The weaviate-client package is required to use WeaviateVectorStore. "
                    "Install it with `pip install kel[weaviate]`."
                ) from exc
            if url is None:
                # anonymous, self-hosted — the common in-cluster deployment, no key needed
                self._client = weaviate.connect_to_local()
            elif api_key:
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url, auth_credentials=AuthApiKey(api_key)
                )
            else:
                raise ValueError(
                    "api_key is required when connecting to Weaviate via a `url` "
                    "(Weaviate Cloud always requires auth); omit `url` instead for "
                    "anonymous self-hosted access via connect_to_local()."
                )

    def _ensure_collection(self) -> Any:
        if not self._collection_ready:
            if not self._client.collections.exists(self.collection_name):
                from weaviate.classes.config import Configure

                self._client.collections.create(
                    self.collection_name, vectorizer_config=Configure.Vectorizer.none()
                )
            self._collection_ready = True
        return self._client.collections.get(self.collection_name)

    def upsert(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        collection = self._ensure_collection()
        for chunk in embedded:
            collection.data.insert(
                uuid=_object_id(chunk.id),
                properties={"chunk_id": chunk.id, "text": chunk.text, **chunk.metadata},
                vector=chunk.embedding,
            )

    def query(self, embedding: list[float], k: int = 5) -> list[ScoredChunk]:
        if not self._collection_ready:
            return []
        from weaviate.classes.query import MetadataQuery

        collection = self._client.collections.get(self.collection_name)
        result = collection.query.near_vector(near_vector=embedding, limit=k, return_metadata=MetadataQuery(distance=True))
        return [
            ScoredChunk(chunk=self._to_chunk(obj.properties), score=1.0 - (obj.metadata.distance or 0.0))
            for obj in result.objects
        ]

    def keyword_query(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not self._collection_ready:
            return []
        collection = self._client.collections.get(self.collection_name)
        result = collection.query.bm25(query=query, limit=k)
        return [ScoredChunk(chunk=self._to_chunk(obj.properties), score=1.0) for obj in result.objects]

    def get(self, chunk_id: str) -> Chunk | None:
        if not self._collection_ready:
            return None
        collection = self._client.collections.get(self.collection_name)
        obj = collection.query.fetch_object_by_id(_object_id(chunk_id))
        if obj is None:
            return None
        return self._to_chunk(obj.properties)

    def delete(self, ids: list[str]) -> None:
        if not self._collection_ready or not ids:
            return
        collection = self._client.collections.get(self.collection_name)
        for chunk_id in ids:
            collection.data.delete_by_id(_object_id(chunk_id))

    @staticmethod
    def _to_chunk(properties: dict[str, Any]) -> Chunk:
        metadata = {k: v for k, v in properties.items() if k not in ("chunk_id", "text")}
        return Chunk(id=properties["chunk_id"], text=properties.get("text", ""), metadata=metadata)
