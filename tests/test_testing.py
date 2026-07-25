import tempfile
from pathlib import Path

import pytest

from helpers import ScriptedModel
from kel.budget import Budget, BudgetTracker
from kel.models.types import Message, ModelResponse, TextPart, Usage
from kel.observability.types import Span
from kel.testing import (
    Cassette,
    RecordingChatModel,
    ReplayChatModel,
    ReplayExhaustedError,
    assert_budget_never_exceeded,
    assert_no_error_spans,
    assert_node_sequence,
    assert_nodes_visited,
    assert_span_names,
)


def _response(text: str, rid: str = "r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage(input_tokens=1, output_tokens=1))


def test_recording_chat_model_captures_calls_into_cassette():
    scripted = ScriptedModel("fake-1", [_response("hello"), _response("world")])
    recorder = RecordingChatModel(scripted)

    r1 = recorder.generate([Message.user("hi")])
    r2 = recorder.generate([Message.user("again")])

    assert r1.text == "hello"
    assert r2.text == "world"
    assert len(recorder.cassette.interactions) == 2
    assert recorder.cassette.interactions[0].request["messages"][0]["content"][0]["text"] == "hi"


def test_cassette_roundtrips_through_disk():
    scripted = ScriptedModel("fake-1", [_response("hello")])
    recorder = RecordingChatModel(scripted)
    recorder.generate([Message.user("hi")])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cassette.json"
        recorder.cassette.save(path)
        loaded = Cassette.load(path)

    assert len(loaded.interactions) == 1
    assert loaded.interactions[0].response.text == "hello"


def test_replay_chat_model_returns_recorded_responses_in_order():
    cassette = Cassette()
    scripted = ScriptedModel("fake-1", [_response("first"), _response("second")])
    recorder = RecordingChatModel(scripted, cassette)
    recorder.generate([Message.user("a")])
    recorder.generate([Message.user("b")])

    replay = ReplayChatModel(cassette)
    assert replay.generate([Message.user("anything")]).text == "first"
    assert replay.generate([Message.user("anything")]).text == "second"


def test_replay_chat_model_raises_when_exhausted():
    cassette = Cassette()
    replay = ReplayChatModel(cassette)
    with pytest.raises(ReplayExhaustedError):
        replay.generate([Message.user("hi")])


def test_assert_node_sequence_passes_and_fails_correctly():
    assert_node_sequence(["a", "b"], ["a", "b"])
    with pytest.raises(AssertionError):
        assert_node_sequence(["a", "b"], ["a", "c"])


def test_assert_nodes_visited_detects_missing_node():
    assert_nodes_visited(["a", "b", "c"], {"a", "c"})
    with pytest.raises(AssertionError):
        assert_nodes_visited(["a", "b"], {"a", "z"})


def test_assert_span_names_and_no_error_spans():
    spans = [
        Span(id="1", name="kel.model.generate", start_time=0, duration_ms=1, status="ok"),
        Span(id="2", name="kel.runtime.node", start_time=0, duration_ms=1, status="ok"),
    ]
    assert_span_names(spans, ["kel.model.generate", "kel.runtime.node"])
    assert_no_error_spans(spans)

    spans_with_error = spans + [Span(id="3", name="broken", start_time=0, duration_ms=1, status="error", error="boom")]
    with pytest.raises(AssertionError):
        assert_no_error_spans(spans_with_error)


def test_assert_budget_never_exceeded_catches_overrun():
    tracker = BudgetTracker(Budget(max_tokens=100))
    tracker.tokens_used = 150  # simulate an overrun bypassing enforcement, as if checked after the fact
    with pytest.raises(AssertionError):
        assert_budget_never_exceeded(tracker)


def test_assert_budget_never_exceeded_passes_within_budget():
    tracker = BudgetTracker(Budget(max_tokens=100))
    tracker.tokens_used = 50
    assert_budget_never_exceeded(tracker)
