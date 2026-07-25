import pytest

from kel.models.base import EmbeddingModel
from kel.retrieval import (
    Chunk,
    InMemoryVectorStore,
    NaiveHashEmbedder,
    Retriever,
    embedder_from_model,
    recursive_split_text,
    split_text,
)


def test_split_text_returns_whole_text_if_under_chunk_size():
    assert split_text("short text", chunk_size=500) == ["short text"]


def test_split_text_returns_empty_list_for_empty_string():
    assert split_text("", chunk_size=500) == []


def test_split_text_produces_overlapping_windows():
    text = "".join(str(i % 10) for i in range(1000))
    chunk_size, overlap = 300, 50
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    assert all(len(c) <= chunk_size for c in chunks)
    assert len(chunks) > 1
    # the tail of chunk[0] should reappear at the head of chunk[1] (the overlap region)
    assert chunks[0][-overlap:] == chunks[1][:overlap]


def test_split_text_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        split_text("x" * 100, chunk_size=50, overlap=50)


def test_naive_hash_embedder_is_deterministic_within_process():
    embedder = NaiveHashEmbedder(dims=32)
    v1 = embedder("hello world")
    v2 = embedder("hello world")
    assert v1 == v2
    assert len(v1) == 32


def test_vector_store_upsert_query_delete_roundtrip():
    store = InMemoryVectorStore()
    store.upsert([Chunk(id="a", text="cats are great", embedding=[1.0, 0.0])])
    store.upsert([Chunk(id="b", text="dogs are great", embedding=[0.0, 1.0])])

    results = store.query([1.0, 0.0], k=1)
    assert results[0].chunk.id == "a"

    assert store.get("a") is not None
    store.delete(["a"])
    assert store.get("a") is None
    assert len(store) == 1


def test_retriever_ingest_and_retrieve_finds_relevant_chunk():
    store = InMemoryVectorStore()
    retriever = Retriever(store, embedder=NaiveHashEmbedder(dims=64))

    retriever.ingest("The quick brown fox jumps over the lazy dog.", id_prefix="doc1")
    retriever.ingest("Paris is the capital of France.", id_prefix="doc2")

    results = retriever.retrieve("capital of France", k=1)
    assert "France" in results[0].chunk.text


def test_retriever_hybrid_search_blends_vector_and_keyword_signals():
    store = InMemoryVectorStore()
    retriever = Retriever(store, embedder=NaiveHashEmbedder(dims=64))

    retriever.ingest("kel is a universal agentic OS for building agents", id_prefix="doc1")
    retriever.ingest("older orchestration frameworks bolt observability on as an afterthought", id_prefix="doc2")

    results = retriever.retrieve_hybrid("agentic OS kel", k=2)
    assert len(results) > 0
    assert "kel" in results[0].chunk.text


class _FakeEmbeddingModel(EmbeddingModel):
    provider = "fake"
    model_id = "fake-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


def test_embedder_from_model_bridges_batch_api_to_single_text_callable():
    model = _FakeEmbeddingModel()
    embed = embedder_from_model(model)

    vector = embed("hello")

    assert vector == [5.0, 0.0]
    assert model.calls == [["hello"]]


def test_embedder_from_model_works_with_retriever():
    model = _FakeEmbeddingModel()
    retriever = Retriever(InMemoryVectorStore(), embedder=embedder_from_model(model))

    retriever.ingest("hi", id_prefix="doc1")
    results = retriever.retrieve("hi", k=1)

    assert len(results) == 1


def test_recursive_split_text_prefers_paragraph_boundaries():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = recursive_split_text(text, chunk_size=30, overlap=0)
    # each chunk should end at a natural boundary, not mid-word
    for chunk in chunks:
        assert chunk == chunk.strip()
    assert "".join(c.replace("\n\n", "") for c in chunks).count("paragraph") == 3


def test_recursive_split_text_falls_back_to_sentence_then_word_then_char():
    # one giant paragraph (no \n\n or \n) forces fallback to ". " then " " then hard split
    text = "This is sentence one. This is sentence two. This is sentence three. " * 5
    chunks = recursive_split_text(text, chunk_size=50, overlap=0)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_recursive_split_text_respects_chunk_size_even_for_unsplittable_text():
    text = "x" * 1000  # no separators at all -> must hard-split
    chunks = recursive_split_text(text, chunk_size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) == 1000


def test_recursive_split_text_returns_whole_text_if_under_chunk_size():
    assert recursive_split_text("short", chunk_size=500) == ["short"]


def test_recursive_split_text_empty_string_returns_empty_list():
    assert recursive_split_text("", chunk_size=500) == []


def test_recursive_split_text_applies_overlap_between_chunks():
    text = "paragraph one here.\n\nparagraph two here.\n\nparagraph three here."
    chunks = recursive_split_text(text, chunk_size=25, overlap=5)
    if len(chunks) > 1:
        assert chunks[1].startswith(chunks[0][-5:])


def test_retriever_uses_custom_splitter():
    captured = []

    def fake_splitter(text, *, chunk_size, overlap):
        captured.append((chunk_size, overlap))
        return [text]  # don't actually split, just record the call

    store = InMemoryVectorStore()
    retriever = Retriever(store, embedder=NaiveHashEmbedder(dims=32), splitter=fake_splitter)
    retriever.ingest("some text", chunk_size=123, overlap=7)

    assert captured == [(123, 7)]
