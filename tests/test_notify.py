import json

import pytest

from kel.runtime import Graph, Interrupt, WebhookNotifier, notify_interrupt, run_graph


def _paused_run():
    graph = Graph(entry="approve")

    def approve(state):
        if "__resume_value__" not in state:
            raise Interrupt({"action": "approve_payment", "amount": 100})
        return {"approved": state["__resume_value__"]}

    graph.add_node("approve", approve)
    graph.set_finish("approve")

    return run_graph(graph, {}, run_id="run-1")


class FakePostSink:
    def __init__(self):
        self.calls: list[tuple[str, bytes]] = []

    def post(self, url: str, payload: bytes) -> None:
        self.calls.append((url, payload))


def test_webhook_notifier_posts_message_and_metadata_as_json():
    sink = FakePostSink()
    notifier = WebhookNotifier("https://example.com/webhook", post=sink.post)

    notifier.notify("hello", metadata={"run_id": "run-1"})

    assert len(sink.calls) == 1
    url, payload = sink.calls[0]
    assert url == "https://example.com/webhook"
    assert json.loads(payload) == {"message": "hello", "metadata": {"run_id": "run-1"}}


def test_webhook_notifier_default_post_rejects_non_http_schemes():
    from kel.runtime.notify import _default_post

    with pytest.raises(ValueError, match="unsupported webhook URL scheme"):
        _default_post("file:///etc/passwd", b"{}")


def test_notify_interrupt_sends_a_message_identifying_the_paused_run():
    paused = _paused_run()
    sink = FakePostSink()
    notifier = WebhookNotifier("https://example.com/webhook", post=sink.post)

    notify_interrupt(paused, notifier)

    assert len(sink.calls) == 1
    _, payload = sink.calls[0]
    body = json.loads(payload)
    assert "run-1" in body["message"]
    assert "approve" in body["message"]
    assert body["metadata"]["run_id"] == "run-1"
    assert body["metadata"]["pending_node"] == "approve"
    assert body["metadata"]["payload"] == {"action": "approve_payment", "amount": 100}


def test_notify_interrupt_accepts_a_custom_message():
    paused = _paused_run()
    sink = FakePostSink()
    notifier = WebhookNotifier("https://example.com/webhook", post=sink.post)

    notify_interrupt(paused, notifier, message="custom notification text")

    body = json.loads(sink.calls[0][1])
    assert body["message"] == "custom notification text"


def test_notify_interrupt_raises_if_the_run_is_not_actually_interrupted():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"done": True})
    graph.set_finish("a")
    completed_run = run_graph(graph, {})

    with pytest.raises(ValueError, match="not interrupted"):
        notify_interrupt(completed_run, WebhookNotifier("https://example.com"))
