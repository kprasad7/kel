import tempfile
from pathlib import Path

from kel.memory import (
    FileEpisodicStore,
    InMemoryEpisodicStore,
    Memory,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
    consolidate,
)
from kel.models.types import Message


def test_working_memory_is_a_context_window():
    wm = WorkingMemory(max_tokens=1000)
    wm.add(Message.user("hi"))
    assert wm.tokens_used > 0


def test_in_memory_episodic_store_roundtrip():
    store = InMemoryEpisodicStore()
    store.append("s1", Message.user("hello"))
    store.append("s1", Message.assistant("hi"))
    store.append("s2", Message.user("other session"))

    assert [m.text for m in store.transcript("s1")] == ["hello", "hi"]
    assert [m.text for m in store.transcript("s2")] == ["other session"]
    assert store.transcript("missing") == []


def test_file_episodic_store_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        store1 = FileEpisodicStore(tmp)
        store1.append("s1", Message.user("hello"))
        store1.append("s1", Message.assistant("hi"))

        store2 = FileEpisodicStore(tmp)
        transcript = store2.transcript("s1")
        assert [m.text for m in transcript] == ["hello", "hi"]
        assert Path(tmp, "s1.jsonl").exists()


def test_semantic_memory_keyword_search_without_embedder():
    sm = SemanticMemory()
    sm.remember("the user prefers dark mode in the UI")
    sm.remember("the deployment pipeline uses GitHub Actions")

    results = sm.search("dark mode preference")
    assert results[0].text == "the user prefers dark mode in the UI"


def test_semantic_memory_with_embedder_uses_cosine_similarity():
    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0] if "cat" in text else [0.0, 1.0]

    sm = SemanticMemory(embedder=fake_embed)
    sm.remember("I have a cat")
    sm.remember("I have a dog")

    results = sm.search("tell me about the cat", k=1)
    assert results[0].text == "I have a cat"


def test_semantic_memory_forget_removes_fact():
    sm = SemanticMemory()
    fid = sm.remember("temporary fact")
    assert len(sm) == 1
    sm.forget(fid)
    assert len(sm) == 0


def test_procedural_memory_save_load_list_delete():
    with tempfile.TemporaryDirectory() as tmp:
        pm = ProceduralMemory(tmp)
        pm.save("retry-pattern", "# Retry pattern\nAlways retry idempotent calls once.")
        assert pm.load("retry-pattern").startswith("# Retry pattern")
        assert pm.list() == ["retry-pattern"]
        pm.delete("retry-pattern")
        assert pm.load("retry-pattern") is None
        assert pm.list() == []


def test_consolidate_summarizes_transcript_into_semantic_memory():
    episodic = InMemoryEpisodicStore()
    episodic.append("s1", Message.user("what's the weather"))
    episodic.append("s1", Message.assistant("it's sunny"))
    semantic = SemanticMemory()

    fact_id = consolidate(episodic, "s1", semantic, summarize=lambda msgs: f"{len(msgs)}-message summary")

    assert fact_id is not None
    facts = semantic.search("summary")
    assert facts[0].text == "2-message summary"
    assert facts[0].metadata["session_id"] == "s1"


def test_consolidate_returns_none_for_empty_session():
    episodic = InMemoryEpisodicStore()
    semantic = SemanticMemory()
    assert consolidate(episodic, "empty", semantic, summarize=lambda msgs: "x") is None


def test_memory_facade_remember_turn_updates_working_and_episodic():
    memory = Memory(session_id="s1")
    memory.remember_turn(Message.user("hello"))
    memory.remember_turn(Message.assistant("hi"))

    assert [m.text for m in memory.working.messages] == ["hello", "hi"]
    assert [m.text for m in memory.episodic.transcript("s1")] == ["hello", "hi"]
