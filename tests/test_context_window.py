import kel.context.eviction as eviction_module
import kel.context.window as window_module
from kel.context import ContextWindow, make_summarization_eviction, sliding_window_eviction
from kel.models.types import Message


def _long_message(role_fn, n_chars: int) -> Message:
    return role_fn("x" * n_chars)


def test_add_within_budget_keeps_all_messages():
    window = ContextWindow(max_tokens=1000)
    window.add(Message.user("hello"))
    window.add(Message.assistant("hi there"))
    assert len(window.messages) == 2
    assert window.evicted_count == 0


def test_sliding_window_evicts_oldest_first():
    # each message ~ (400 chars // 4) + 4 = 104 tokens; budget of 250 keeps ~2
    window = ContextWindow(max_tokens=250, policy=sliding_window_eviction)
    for _i in range(5):
        window.add(_long_message(Message.user, 400))
    assert window.tokens_used <= 250
    assert len(window.messages) < 5
    assert window.evicted_count > 0


def test_summarization_eviction_replaces_old_messages_with_summary():
    def fake_summarize(messages: list[Message]) -> Message:
        return Message.assistant(f"summary of {len(messages)} messages")

    policy = make_summarization_eviction(fake_summarize, keep_recent=2)
    window = ContextWindow(max_tokens=150, policy=policy)
    for _i in range(6):
        window.add(_long_message(Message.user, 200))

    assert window.tokens_used <= 150 or len(window.messages) <= 3
    assert any("summary of" in m.text for m in window.messages)
    # most recent messages are preserved verbatim
    assert window.messages[-1].text == "x" * 200


def test_tokens_remaining_reflects_usage():
    window = ContextWindow(max_tokens=1000)
    window.add(Message.user("hi"))
    assert window.tokens_remaining == 1000 - window.tokens_used


def test_add_maintains_a_running_total_instead_of_rescanning_full_history(monkeypatch):
    # tokens_used must be O(1) per add() — Memory.working is a
    # ContextWindow, and every Agent.run() turn calls add(), so rescanning
    # the whole history on every call would make an n-turn session cost
    # O(n^2). Assert this by counting estimate_total_tokens calls, which
    # should only fire once (at construction) — not once per add().
    calls = []
    original = window_module.estimate_total_tokens

    def counting_wrapper(messages):
        calls.append(len(messages))
        return original(messages)

    monkeypatch.setattr(window_module, "estimate_total_tokens", counting_wrapper)

    window = ContextWindow(max_tokens=10_000)  # high enough that eviction never triggers
    assert len(calls) == 1  # the constructor's initial computation

    for i in range(20):
        window.add(Message.user(f"message {i}"))

    assert len(calls) == 1  # still just the one call from construction — no per-add rescans
    assert window.tokens_used > 0


def test_sliding_window_eviction_computes_total_once_regardless_of_how_many_are_evicted(monkeypatch):
    # evicting k of n messages must be O(n) total, not O(k*n): assert
    # estimate_total_tokens is only ever called once (up front), with each
    # subsequent pop just decrementing a running total.
    calls = []
    original = eviction_module.estimate_total_tokens

    def counting_wrapper(messages):
        calls.append(len(messages))
        return original(messages)

    monkeypatch.setattr(eviction_module, "estimate_total_tokens", counting_wrapper)

    messages = [_long_message(Message.user, 400) for _ in range(20)]
    kept = sliding_window_eviction(messages, max_tokens=250)

    assert len(calls) == 1
    assert len(kept) < len(messages)
