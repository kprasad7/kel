"""Chroma adapter for `kel.retrieval.VectorStore`. Requires
`pip install kel[chroma]`.

Same documented-shape caveat as the other real-vendor adapters.

**No credentials required for the default local/embedded modes** —
`path=None` (default) uses an in-memory ephemeral client; passing `path`
uses `PersistentClient` (a local on-disk store, still no auth, the common
choice for a sidecar/local deployment). `host` switches to `HttpClient`
for a real Chroma server, with `token` optional (only needed if that
server has auth enabled — many in-cluster deployments don't).

Chroma accepts arbitrary string IDs natively — no UUID-mapping workaround
needed here, unlike Qdrant/Weaviate.
"""

from __future__ import annotations

from typing import Any

from kel.retrieval.types import Chunk, ScoredChunk


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        *,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        token: str | None = None,
        client: Any = None,
    ):
        self.collection_name = collection_name
        if client is not None:
            self._client = client
        else:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "The chromadb package is required to use ChromaVectorStore. "
                    "Install it with `pip install kel[chroma]`."
                ) from exc
            if host is not None:
                headers = {"Authorization": f"Bearer {token}"} if token else None
                self._client = chromadb.HttpClient(host=host, port=port, headers=headers)
            elif path is not None:
                self._client = chromadb.PersistentClient(path=path)
            else:
                self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        self._collection.upsert(
            ids=[c.id for c in embedded],
            embeddings=[c.embedding for c in embedded],
            documents=[c.text for c in embedded],
            metadatas=[c.metadata or {} for c in embedded],
        )

    def query(self, embedding: list[float], k: int = 5) -> list[ScoredChunk]:
        result = self._collection.query(query_embeddings=[embedding], n_results=k)
        ids = result["ids"][0] if result["ids"] else []
        docs = result["documents"][0] if result.get("documents") else [""] * len(ids)
        metas = result["metadatas"][0] if result.get("metadatas") else [{}] * len(ids)
        distances = result["distances"][0] if result.get("distances") else [0.0] * len(ids)
        return [
            ScoredChunk(chunk=Chunk(id=i, text=d or "", metadata=m or {}), score=1.0 - dist)
            for i, d, m, dist in zip(ids, docs, metas, distances)
        ]

    def keyword_query(self, query: str, k: int = 5) -> list[ScoredChunk]:
        result = self._collection.get(where_document={"$contains": query}, limit=k)
        ids = result.get("ids", [])
        docs = result.get("documents") or [""] * len(ids)
        metas = result.get("metadatas") or [{}] * len(ids)
        return [
            ScoredChunk(chunk=Chunk(id=i, text=d or "", metadata=m or {}), score=1.0)
            for i, d, m in zip(ids, docs, metas)
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        result = self._collection.get(ids=[chunk_id])
        ids = result.get("ids", [])
        if not ids:
            return None
        docs = result.get("documents") or [""]
        metas = result.get("metadatas") or [{}]
        return Chunk(id=ids[0], text=docs[0] or "", metadata=metas[0] or {})

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)
