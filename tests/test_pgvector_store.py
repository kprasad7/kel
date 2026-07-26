import json

from kel.retrieval.pgvector_store import PgVectorStore
from kel.retrieval.types import Chunk


def _matches_metadata_filter(metadata, filter_pairs):
    # mirrors what Postgres's `metadata->>%s = %s` actually compares:
    # the JSON value extracted as text, so a stored int/bool must be
    # compared via its JSON text form, not Python's str().
    for key, value in zip(filter_pairs[0::2], filter_pairs[1::2], strict=True):
        actual = metadata.get(key)
        actual_text = actual if isinstance(actual, str) else json.dumps(actual)
        if actual_text != value:
            return False
    return True


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
            # params: (embedding, *filter_pairs, embedding, k) — filter
            # pairs (if any) sit between the two embedding params.
            filter_pairs = params[1:-2]
            rows = [r for r in self.store.rows.values() if _matches_metadata_filter(r[2], filter_pairs)]
            self._last_result = [(r[0], r[1], r[2], 0.95) for r in rows]
        elif "ILIKE" in sql_upper:
            # params: (needle, *filter_pairs, k)
            needle = params[0].strip("%").lower()
            filter_pairs = params[1:-1]
            rows = [r for r in self.store.rows.values() if needle in r[1].lower()]
            rows = [r for r in rows if _matches_metadata_filter(r[2], filter_pairs)]
            self._last_result = [r[:3] for r in rows]
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


def test_query_scopes_results_to_matching_metadata():
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)
    store.upsert(
        [
            Chunk(id="doc1", text="a", metadata={"user_id": "u1"}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="b", metadata={"user_id": "u2"}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.query([1.0, 0.0], k=5, filter={"user_id": "u1"})

    assert [r.chunk.id for r in results] == ["doc1"]


def test_keyword_query_scopes_results_to_matching_metadata():
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)
    store.upsert(
        [
            Chunk(id="doc1", text="agentic OS", metadata={"user_id": "u1"}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="agentic OS", metadata={"user_id": "u2"}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.keyword_query("agentic", k=5, filter={"user_id": "u2"})

    assert [r.chunk.id for r in results] == ["doc2"]


def test_filter_matches_non_string_metadata_values_correctly():
    # non-string values are compared via their JSON text form (what
    # Postgres's `->>` operator actually returns), not Python's str()
    conn = FakeConnection()
    store = PgVectorStore("test_table", connection=conn)
    store.upsert(
        [
            Chunk(id="doc1", text="a", metadata={"is_active": True, "rank": 1}, embedding=[1.0, 0.0]),
            Chunk(id="doc2", text="b", metadata={"is_active": False, "rank": 2}, embedding=[1.0, 0.0]),
        ]
    )

    results = store.query([1.0, 0.0], k=5, filter={"is_active": True, "rank": 1})

    assert [r.chunk.id for r in results] == ["doc1"]
