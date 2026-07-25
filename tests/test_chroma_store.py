from kel.retrieval.chroma_store import ChromaVectorStore
from kel.retrieval.types import Chunk


class FakeCollection:
    def __init__(self):
        self._data: dict[str, dict] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            self._data[i] = {"embedding": e, "document": d, "metadata": m}

    def query(self, query_embeddings, n_results):
        items = list(self._data.items())[:n_results]
        return {
            "ids": [[i for i, _ in items]],
            "documents": [[d["document"] for _, d in items]],
            "metadatas": [[d["metadata"] for _, d in items]],
            "distances": [[0.1 for _ in items]],
        }

    def get(self, ids=None, where_document=None, limit=None):
        if ids is not None:
            matches = {i: self._data[i] for i in ids if i in self._data}
        elif where_document is not None:
            needle = where_document["$contains"].lower()
            matches = {i: d for i, d in self._data.items() if needle in d["document"].lower()}
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

    def get_or_create_collection(self, name):
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
