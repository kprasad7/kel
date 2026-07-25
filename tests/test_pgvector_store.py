import json

from kel.retrieval.pgvector_store import PgVectorStore
from kel.retrieval.types import Chunk


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("CREATE"):
            return
        if sql_upper.startswith("INSERT"):
            chunk_id, text, metadata_json, embedding = params
            self.store.rows[chunk_id] = (chunk_id, text, json.loads(metadata_json), embedding)
        elif sql_upper.startswith("SELECT ID, TEXT, METADATA, 1"):
            embedding = params[0]
            self._last_result = [
                (r[0], r[1], r[2], 0.95) for r in list(self.store.rows.values())
            ]
        elif "ILIKE" in sql_upper:
            needle = params[0].strip("%").lower()
            self._last_result = [r[:3] for r in self.store.rows.values() if needle in r[1].lower()]
        elif sql_upper.startswith("SELECT ID, TEXT, METADATA FROM") and "WHERE ID = %S" in sql_upper:
            chunk_id = params[0]
            row = self.store.rows.get(chunk_id)
            self._last_result = row[:3] if row else None
        elif sql_upper.startswith("DELETE"):
            for chunk_id in params[0]:
                self.store.rows.pop(chunk_id, None)

    def fetchall(self):
        return self._last_result or []

    def fetchone(self):
        return self._last_result


class FakeConnection:
    def __init__(self):
        self.rows: dict[str, tuple] = {}

    def cursor(self):
        return FakeCursor(self)


def test_upsert_and_query_roundtrip():
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)

    store.upsert([Chunk(id="doc1", text="hello world", metadata={"source": "a"}, embedding=[1.0, 0.0])])
    results = store.query([1.0, 0.0], k=5)

    assert len(results) == 1
    assert results[0].chunk.id == "doc1"
    assert results[0].chunk.text == "hello world"
    assert results[0].chunk.metadata == {"source": "a"}
    assert results[0].score == 0.95


def test_get_and_delete_roundtrip():
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)
    store.upsert([Chunk(id="doc1", text="hello", embedding=[1.0, 0.0])])

    assert store.get("doc1").text == "hello"
    store.delete(["doc1"])
    assert store.get("doc1") is None


def test_keyword_query_matches_via_ilike():
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)
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
    store = PgVectorStore("test_table", connection=FakeConnection())
    assert store.query([1.0, 0.0]) == []
    assert store.get("x") is None
