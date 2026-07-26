import tempfile
import time
from pathlib import Path

from kel.memory import (
    FileEpisodicStore,
    InMemoryEpisodicStore,
    Memory,
    ProceduralMemory,
    SemanticFact,
    SemanticMemory,
    SQLiteEpisodicStore,
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


def test_sqlite_episodic_store_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp, "episodic.sqlite")
        store1 = SQLiteEpisodicStore(db_path)
        store1.append("s1", Message.user("hello"))
        store1.append("s1", Message.assistant("hi"))
        store1.append("s2", Message.user("other session"))

        store2 = SQLiteEpisodicStore(db_path)
        assert [m.text for m in store2.transcript("s1")] == ["hello", "hi"]
        assert [m.text for m in store2.transcript("s2")] == ["other session"]
        assert store2.transcript("missing") == []
        assert sorted(store2.sessions()) == ["s1", "s2"]

        store1.close()
        store2.close()


def test_sqlite_episodic_store_preserves_message_order_within_a_session():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEpisodicStore(Path(tmp, "episodic.sqlite"))
        for i in range(10):
            store.append("s1", Message.user(f"turn {i}"))

        transcript = store.transcript("s1")
        assert [m.text for m in transcript] == [f"turn {i}" for i in range(10)]

        store.close()


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


def test_semantic_fact_without_ttl_never_expires():
    fact = SemanticFact(id="1", text="foundational fact", created_at=0.0, ttl_seconds=None)
    assert fact.is_expired(now=10_000_000.0) is False


def test_semantic_fact_with_ttl_expires_after_the_deadline():
    fact = SemanticFact(id="1", text="passing remark", created_at=1000.0, ttl_seconds=60.0)
    assert fact.is_expired(now=1030.0) is False  # 30s elapsed, still within 60s TTL
    assert fact.is_expired(now=1061.0) is True  # 61s elapsed, past the 60s TTL


def test_search_excludes_expired_facts():
    sm = SemanticMemory()
    sm.remember("permanent fact about dark mode preference", id="permanent")
    sm.remember("temporary fact about dark mode preference", id="temporary", ttl_seconds=1.0)
    # force the temporary fact into the past so it's already expired
    sm._facts["temporary"].created_at = time.time() - 100

    results = sm.search("dark mode")

    assert [f.id for f in results] == ["permanent"]


def test_forget_expired_purges_only_expired_facts_and_returns_the_count():
    sm = SemanticMemory()
    sm.remember("permanent fact", id="permanent")
    sm.remember("temporary fact", id="temporary", ttl_seconds=1.0)
    sm._facts["temporary"].created_at = time.time() - 100

    removed = sm.forget_expired()

    assert removed == 1
    assert len(sm) == 1
    assert sm.search("permanent")[0].id == "permanent"


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


def test_memory_recalls_prior_turns_when_reconstructed_with_the_same_session():
    # a Memory (or an Agent built on one) reconstructed fresh — e.g. every
    # rerun of a Streamlit-style script, or a new process — should resume
    # a session instead of silently starting over, as long as the caller
    # passes the same durable episodic store and session_id back in.
    with tempfile.TemporaryDirectory() as tmp:
        episodic = FileEpisodicStore(tmp)

        first = Memory(session_id="s1", episodic=episodic)
        first.remember_turn(Message.user("what's the capital of France?"))
        first.remember_turn(Message.assistant("Paris"))

        # simulates a fresh process/script rerun: brand new Memory object,
        # same durable store + session_id
        second = Memory(session_id="s1", episodic=episodic)

        assert [m.text for m in second.working.messages] == ["what's the capital of France?", "Paris"]


def test_memory_starts_fresh_for_a_new_session_id_even_with_a_shared_store():
    with tempfile.TemporaryDirectory() as tmp:
        episodic = FileEpisodicStore(tmp)
        first = Memory(session_id="s1", episodic=episodic)
        first.remember_turn(Message.user("hello"))

        other_session = Memory(session_id="s2", episodic=episodic)

        assert other_session.working.messages == []


def test_memory_does_not_recall_across_instances_with_the_default_in_memory_store():
    # InMemoryEpisodicStore doesn't outlive the process — a fresh Memory()
    # with no episodic store passed in has nothing to recall from,
    # matching the pre-existing default behavior.
    first = Memory(session_id="s1")
    first.remember_turn(Message.user("hello"))

    second = Memory(session_id="s1")

    assert second.working.messages == []


def test_memory_recalls_prior_turns_via_sqlite_episodic_store_across_processes():
    # same recall contract as FileEpisodicStore, but backed by SQLite —
    # this is the option that also works from multiple worker processes
    # sharing one file, not just multiple in-process reconstructions.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp, "episodic.sqlite")
        episodic = SQLiteEpisodicStore(db_path)

        first = Memory(session_id="s1", episodic=episodic)
        first.remember_turn(Message.user("what's the capital of France?"))
        first.remember_turn(Message.assistant("Paris"))

        # a different SQLiteEpisodicStore instance pointed at the same
        # file simulates a separate worker process
        reopened = SQLiteEpisodicStore(db_path)
        second = Memory(session_id="s1", episodic=reopened)

        assert [m.text for m in second.working.messages] == ["what's the capital of France?", "Paris"]

        episodic.close()
        reopened.close()
