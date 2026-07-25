import pytest

from kel import Message, MessageStop, TextDelta, TextPart, Usage
from kel.models.base import ChatModel
from kel.models.types import ModelResponse
from kel.observability import InstrumentedChatModel, ListSink, Tracer


class _FakeModel(ChatModel):
    provider = "fake"
    model_id = "fake-1"

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def generate(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._response

    def stream(self, messages, **kwargs):
        if self._error:
            raise self._error
        yield TextDelta(text="hi")
        yield MessageStop(response=self._response)

    async def agenerate(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._response

    async def astream(self, messages, **kwargs):
        if self._error:
            raise self._error
        yield TextDelta(text="hi")
        yield MessageStop(response=self._response)


def _response():
    return ModelResponse(
        id="r1",
        model="fake-1",
        content=[TextPart(text="hello")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=3, output_tokens=2),
    )


def test_generate_delegates_and_records_span():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    wrapped = _FakeModel(response=_response())
    model = InstrumentedChatModel(wrapped, tracer=tracer)

    resp = model.generate([Message.user("hi")])

    assert resp.text == "hello"
    span = sink.spans[0]
    assert span.name == "kel.model.generate"
    assert span.attributes["provider"] == "fake"
    assert span.attributes["model"] == "fake-1"
    assert span.attributes["input_tokens"] == 3
    assert span.attributes["output_tokens"] == 2
    assert span.attributes["stop_reason"] == "end_turn"
    assert span.status == "ok"


def test_generate_error_records_error_span_and_reraises():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    wrapped = _FakeModel(error=RuntimeError("provider down"))
    model = InstrumentedChatModel(wrapped, tracer=tracer)

    with pytest.raises(RuntimeError):
        model.generate([Message.user("hi")])

    assert sink.spans[0].status == "error"
    assert sink.spans[0].error == "provider down"


def test_stream_delegates_events_and_records_span_on_completion():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    wrapped = _FakeModel(response=_response())
    model = InstrumentedChatModel(wrapped, tracer=tracer)

    events = list(model.stream([Message.user("hi")]))

    assert isinstance(events[0], TextDelta)
    assert isinstance(events[-1], MessageStop)
    span = sink.spans[0]
    assert span.name == "kel.model.stream"
    assert span.attributes["output_tokens"] == 2


async def test_agenerate_delegates_and_records_span():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    wrapped = _FakeModel(response=_response())
    model = InstrumentedChatModel(wrapped, tracer=tracer)

    resp = await model.agenerate([Message.user("hi")])

    assert resp.text == "hello"
    assert sink.spans[0].name == "kel.model.agenerate"


async def test_astream_delegates_events_and_records_span():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])
    wrapped = _FakeModel(response=_response())
    model = InstrumentedChatModel(wrapped, tracer=tracer)

    events = [event async for event in model.astream([Message.user("hi")])]

    assert isinstance(events[0], TextDelta)
    assert isinstance(events[-1], MessageStop)
    span = sink.spans[0]
    assert span.name == "kel.model.astream"
    assert span.attributes["output_tokens"] == 2
