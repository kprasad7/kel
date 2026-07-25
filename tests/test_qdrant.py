from types import SimpleNamespace

from kel.retrieval.qdrant import QdrantVectorStore
from kel.retrieval.types import Chunk


class FakeQdrantClient:
    """Mimics qdrant_client.QdrantClient's documented API surface just
    enough to test kel's translation logic (id mapping, payload shape),
    without needing a real Qdrant server."""

    def __init__(self):
        self.collections: set[str] = set()
        self.points: dict[str, dict[str, dict]] = {}  # collection -> point_id -> {"vector":..., "payload":...}
        self.payload_indexes: list[tuple[str, str]] = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)
        self.points[collection_name] = {}

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.payload_indexes.append((collection_name, field_name))

    def upsert(self, collection_name, points):
        for point in points:
            self.points[collection_name][point.id] = {"vector": point.vector, "payload": point.payload}

    def query_points(self, collection_name, query, limit):
        items = list(self.points.get(collection_name, {}).items())[:limit]
        points = [SimpleNamespace(id=pid, payload=data["payload"], score=0.9) for pid, data in items]
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, scroll_filter, limit):
        query_text = scroll_filter.must[0].match.text.lower()
        matches = [
            SimpleNamespace(id=pid, payload=data["payload"])
            for pid, data in self.points.get(collection_name, {}).items()
            if query_text in data["payload"].get("text", "").lower()
        ]
        return matches[:limit], None

    def retrieve(self, collection_name, ids):
        return [
            SimpleNamespace(id=pid, payload=self.points[collection_name][pid]["payload"])
            for pid in ids
            if pid in self.points.get(collection_name, {})
        ]

    def delete(self, collection_name, points_selector):
        for pid in points_selector.points:
            self.points[collection_name].pop(pid, None)


def test_upsert_creates_collection_and_stores_points_with_original_chunk_id_in_payload():
    client = FakeQdrantClient()
    store = QdrantVectorStore("test-collection", client=client)

    store.upsert([Chunk(id="doc1-0", text="hello world", metadata={"source": "a"}, embedding=[1.0, 0.0])])

    assert "test-collection" in client.collections
    assert ("test-collection", "text") in client.payload_indexes
    stored = next(iter(client.points["test-collection"].values()))
    assert stored["payload"]["chunk_id"] == "doc1-0"
    assert stored["payload"]["text"] == "hello world"
    assert stored["payload"]["source"] == "a"


def test_query_returns_scored_chunks_with_original_ids():
    client = FakeQdrantClient()
    store = QdrantVectorStore("test-collection", client=client)
    store.upsert([Chunk(id="doc1-0", text="hello world", embedding=[1.0, 0.0])])

    results = store.query([1.0, 0.0], k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1-0"
    assert results[0].chunk.text == "hello world"
    assert results[0].score == 0.9


def test_get_roundtrips_by_original_chunk_id():
    client = FakeQdrantClient()
    store = QdrantVectorStore("test-collection", client=client)
    store.upsert([Chunk(id="doc1-0", text="hello world", embedding=[1.0, 0.0])])

    chunk = store.get("doc1-0")
    assert chunk is not None
    assert chunk.text == "hello world"

    assert store.get("nonexistent") is None


def test_get_before_any_upsert_returns_none():
    store = QdrantVectorStore("test-collection", client=FakeQdrantClient())
    assert store.get("anything") is None


def test_query_before_any_upsert_returns_empty_list():
    store = QdrantVectorStore("test-collection", client=FakeQdrantClient())
    assert store.query([1.0, 0.0]) == []


def test_delete_removes_point_by_original_chunk_id():
    client = FakeQdrantClient()
    store = QdrantVectorStore("test-collection", client=client)
    store.upsert([Chunk(id="doc1-0", text="hello world", embedding=[1.0, 0.0])])

    store.delete(["doc1-0"])

    assert store.get("doc1-0") is None


def test_keyword_query_matches_via_text_filter():
    client = FakeQdrantClient()
    store = QdrantVectorStore("test-collection", client=client)
    store.upsert(
        [
            Chunk(id="doc1", text="kel is a universal agentic OS", embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="completely unrelated content", embedding=[0.0, 1.0]),
        ]
    )

    results = store.keyword_query("agentic", k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"
