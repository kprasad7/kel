from types import SimpleNamespace

from kel.retrieval.weaviate_store import WeaviateVectorStore, _object_id


class FakeData:
    def __init__(self, objects):
        self._objects = objects

    def insert(self, uuid, properties, vector):
        self._objects[uuid] = {"properties": properties, "vector": vector}

    def delete_by_id(self, uuid):
        self._objects.pop(uuid, None)


class FakeQuery:
    def __init__(self, objects):
        self._objects = objects

    def near_vector(self, near_vector, limit, return_metadata):
        items = list(self._objects.items())[:limit]
        return SimpleNamespace(
            objects=[
                SimpleNamespace(properties=data["properties"], metadata=SimpleNamespace(distance=0.1))
                for _, data in items
            ]
        )

    def bm25(self, query, limit):
        matches = [
            SimpleNamespace(properties=data["properties"])
            for data in self._objects.values()
            if query.lower() in data["properties"].get("text", "").lower()
        ][:limit]
        return SimpleNamespace(objects=matches)

    def fetch_object_by_id(self, uuid):
        data = self._objects.get(uuid)
        return SimpleNamespace(properties=data["properties"]) if data else None


class FakeCollection:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.data = FakeData(self.objects)
        self.query = FakeQuery(self.objects)


class FakeCollections:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def exists(self, name):
        return name in self._collections

    def create(self, name, vectorizer_config=None):
        self._collections[name] = FakeCollection()

    def get(self, name):
        if name not in self._collections:
            self._collections[name] = FakeCollection()
        return self._collections[name]


class FakeClient:
    def __init__(self):
        self.collections = FakeCollections()


def test_upsert_creates_collection_and_stores_object_with_original_chunk_id():
    from kel.retrieval.types import Chunk

    client = FakeClient()
    store = WeaviateVectorStore("TestCollection", client=client)

    store.upsert([Chunk(id="doc1-0", text="hello world", metadata={"source": "a"}, embedding=[1.0, 0.0])])

    collection = client.collections.get("TestCollection")
    stored = collection.objects[_object_id("doc1-0")]
    assert stored["properties"]["chunk_id"] == "doc1-0"
    assert stored["properties"]["text"] == "hello world"


def test_query_returns_scored_chunks_with_original_ids():
    from kel.retrieval.types import Chunk

    client = FakeClient()
    store = WeaviateVectorStore("TestCollection", client=client)
    store.upsert([Chunk(id="doc1-0", text="hello world", embedding=[1.0, 0.0])])

    results = store.query([1.0, 0.0], k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1-0"
    assert results[0].score == 0.9  # 1.0 - distance(0.1)


def test_get_and_delete_roundtrip():
    from kel.retrieval.types import Chunk

    client = FakeClient()
    store = WeaviateVectorStore("TestCollection", client=client)
    store.upsert([Chunk(id="doc1-0", text="hello world", embedding=[1.0, 0.0])])

    assert store.get("doc1-0").text == "hello world"
    store.delete(["doc1-0"])
    assert store.get("doc1-0") is None


def test_keyword_query_matches_via_bm25():
    from kel.retrieval.types import Chunk

    client = FakeClient()
    store = WeaviateVectorStore("TestCollection", client=client)
    store.upsert(
        [
            Chunk(id="doc1", text="kel is a universal agentic OS", embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="completely unrelated content", embedding=[0.0, 1.0]),
        ]
    )

    results = store.keyword_query("agentic", k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"


def test_query_before_any_upsert_returns_empty():
    store = WeaviateVectorStore("TestCollection", client=FakeClient())
    assert store.query([1.0, 0.0]) == []
    assert store.get("x") is None
