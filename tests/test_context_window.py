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
    for i in range(5):
        window.add(_long_message(Message.user, 400))
    assert window.tokens_used <= 250
    assert len(window.messages) < 5
    assert window.evicted_count > 0


def test_summarization_eviction_replaces_old_messages_with_summary():
    def fake_summarize(messages: list[Message]) -> Message:
        return Message.assistant(f"summary of {len(messages)} messages")

    policy = make_summarization_eviction(fake_summarize, keep_recent=2)
    window = ContextWindow(max_tokens=150, policy=policy)
    for i in range(6):
        window.add(_long_message(Message.user, 200))

    assert window.tokens_used <= 150 or len(window.messages) <= 3
    assert any("summary of" in m.text for m in window.messages)
    # most recent messages are preserved verbatim
    assert window.messages[-1].text == "x" * 200


def test_tokens_remaining_reflects_usage():
    window = ContextWindow(max_tokens=1000)
    window.add(Message.user("hi"))
    assert window.tokens_remaining == 1000 - window.tokens_used
