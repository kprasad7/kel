from types import SimpleNamespace

from kel.retrieval.pinecone_store import PineconeVectorStore
from kel.retrieval.types import Chunk


class FakeIndex:
    def __init__(self):
        self.vectors: dict[str, dict] = {}

    def upsert(self, vectors):
        for v in vectors:
            self.vectors[v["id"]] = {"values": v["values"], "metadata": v["metadata"]}

    def query(self, vector, top_k, include_metadata=True, filter=None):
        def matches_filter(metadata):
            if not filter:
                return True
            return all(metadata.get(key) == cond["$eq"] for key, cond in filter.items())

        matches = [
            SimpleNamespace(id=i, score=0.95, metadata=d["metadata"])
            for i, d in self.vectors.items()
            if matches_filter(d["metadata"])
        ][:top_k]
        return SimpleNamespace(matches=matches)

    def fetch(self, ids):
        return SimpleNamespace(
            vectors={i: SimpleNamespace(metadata=self.vectors[i]["metadata"]) for i in ids if i in self.vectors}
        )

    def delete(self, ids):
        for i in ids:
            self.vectors.pop(i, None)


def test_upsert_and_query_roundtrip():
    index = FakeIndex()
    store = PineconeVectorStore("test-index", index=index)

    store.upsert([Chunk(id="doc1", text="hello world", metadata={"source": "a"}, embedding=[1.0, 0.0])])
    results = store.query([1.0, 0.0], k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"
    assert results[0].chunk.text == "hello world"
    assert results[0].chunk.metadata == {"source": "a"}
    assert results[0].score == 0.95


def test_get_and_delete():
    index = FakeIndex()
    store = PineconeVectorStore("test-index", index=index)
    store.upsert([Chunk(id="doc1", text="hello", embedding=[1.0, 0.0])])

    assert store.get("doc1").text == "hello"
    store.delete(["doc1"])
    assert store.get("doc1") is None


def test_keyword_query_returns_empty_known_limitation():
    store = PineconeVectorStore("test-index", index=FakeIndex())
    assert store.keyword_query("anything") == []


def test_query_before_upsert_returns_empty_when_no_index():
    store = PineconeVectorStore("test-index", client=SimpleNamespace())
    assert store.query([1.0, 0.0]) == []
    assert store.get("x") is None


def test_query_scopes_results_to_matching_metadata():
    index = FakeIndex()
    store = PineconeVectorStore("test-index", index=index)
    store.upsert(
        [
            Chunk(id="doc1", text="a", metadata={"user_id": "u1"}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="b", metadata={"user_id": "u2"}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.query([1.0, 0.0], k=5, filter={"user_id": "u1"})

    assert [r.chunk.id for r in results] == ["doc1"]
