from kel.retrieval.chroma_store import ChromaVectorStore
from kel.retrieval.types import Chunk


def _matches_where(metadata, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, cond) for cond in where["$and"])
    return all(metadata.get(key) == value for key, value in where.items())


class FakeCollection:
    def __init__(self):
        self._data: dict[str, dict] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            self._data[i] = {"embedding": e, "document": d, "metadata": m}

    def query(self, query_embeddings, n_results, where=None):
        items = [(i, d) for i, d in self._data.items() if _matches_where(d["metadata"], where)][:n_results]
        return {
            "ids": [[i for i, _ in items]],
            "documents": [[d["document"] for _, d in items]],
            "metadatas": [[d["metadata"] for _, d in items]],
            "distances": [[0.1 for _ in items]],
        }

    def get(self, ids=None, where_document=None, where=None, limit=None):
        if ids is not None:
            matches = {i: self._data[i] for i in ids if i in self._data}
        elif where_document is not None:
            needle = where_document["$contains"].lower()
            matches = {
                i: d
                for i, d in self._data.items()
                if needle in d["document"].lower() and _matches_where(d["metadata"], where)
            }
            if limit:
                matches = dict(list(matches.items())[:limit])
        else:
            matches = self._data
        return {
            "ids": list(matches.keys()),
            "documents": [d["document"] for d in matches.values()],
            "metadatas": [d["metadata"] for d in matches.values()],
        }

    def delete(self, ids):
        for i in ids:
            self._data.pop(i, None)


class FakeChromaClient:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}
        self.last_metadata: dict | None = None

    def get_or_create_collection(self, name, metadata=None):
        self.last_metadata = metadata
        return self._collections.setdefault(name, FakeCollection())


def test_upsert_and_query_roundtrip():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert([Chunk(id="doc1", text="hello world", metadata={"source": "a"}, embedding=[1.0, 0.0])])

    results = store.query([1.0, 0.0], k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"
    assert results[0].chunk.text == "hello world"
    assert results[0].chunk.metadata == {"source": "a"}
    assert results[0].score == 0.9


def test_get_and_delete_roundtrip():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert([Chunk(id="doc1", text="hello", embedding=[1.0, 0.0])])

    assert store.get("doc1").text == "hello"
    store.delete(["doc1"])
    assert store.get("doc1") is None


def test_keyword_query_matches_via_where_document_contains():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert(
        [
            Chunk(id="doc1", text="kel is a universal agentic OS", embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="completely unrelated content", embedding=[0.0, 1.0]),
        ]
    )

    results = store.keyword_query("agentic", k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"


def test_query_with_no_data_returns_empty():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    assert store.query([1.0, 0.0]) == []


def test_query_scopes_results_to_matching_metadata():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert(
        [
            Chunk(id="doc1", text="a", metadata={"user_id": "u1"}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="b", metadata={"user_id": "u2"}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.query([1.0, 0.0], k=5, filter={"user_id": "u1"})

    assert [r.chunk.id for r in results] == ["doc1"]


def test_keyword_query_scopes_results_to_matching_metadata():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert(
        [
            Chunk(id="doc1", text="agentic OS", metadata={"user_id": "u1"}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="agentic OS", metadata={"user_id": "u2"}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.keyword_query("agentic", k=5, filter={"user_id": "u2"})

    assert [r.chunk.id for r in results] == ["doc2"]


def test_filter_with_multiple_keys_requires_all_to_match():
    store = ChromaVectorStore("test-collection", client=FakeChromaClient())
    store.upsert(
        [Chunk(id="doc1", text="a", metadata={"user_id": "u1", "source": "amazon"}, embedding=[1.0, 0.0])]
    )
    store.upsert([Chunk(id="doc2", text="b", metadata={"user_id": "u1", "source": "ebay"}, embedding=[1.0, 0.0])])

    results = store.query([1.0, 0.0], k=5, filter={"user_id": "u1", "source": "amazon"})

    assert [r.chunk.id for r in results] == ["doc1"]


def test_collection_is_created_with_cosine_distance_explicitly():
    # Chroma defaults to l2 (squared Euclidean) distance unless told
    # otherwise, which would make query()'s `1.0 - distance` score
    # nonsensical and inconsistent with every other VectorStore adapter's
    # cosine-similarity scores (Qdrant, Pinecone, pgvector).
    client = FakeChromaClient()
    ChromaVectorStore("test-collection", client=client)
    assert client.last_metadata == {"hnsw:space": "cosine"}
