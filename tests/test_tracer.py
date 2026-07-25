import pytest

from kel.observability import ListSink, Tracer


def test_span_records_name_attributes_and_status():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])

    with tracer.span("op", provider="anthropic") as span:
        span.set_attribute("input_tokens", 10)

    assert len(sink.spans) == 1
    recorded = sink.spans[0]
    assert recorded.name == "op"
    assert recorded.status == "ok"
    assert recorded.attributes == {"provider": "anthropic", "input_tokens": 10}
    assert recorded.duration_ms >= 0


def test_span_records_error_status_and_reraises():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])

    with pytest.raises(ValueError):
        with tracer.span("op"):
            raise ValueError("boom")

    recorded = sink.spans[0]
    assert recorded.status == "error"
    assert recorded.error == "boom"


def test_nested_spans_set_parent_id():
    sink = ListSink()
    tracer = Tracer(sinks=[sink])

    with tracer.span("outer") as outer:
        with tracer.span("inner") as inner:
            pass

    inner_span = next(s for s in sink.spans if s.name == "inner")
    outer_span = next(s for s in sink.spans if s.name == "outer")
    assert inner_span.parent_id == outer_span.id
    assert outer_span.parent_id is None


def test_multiple_sinks_all_receive_span():
    sink_a, sink_b = ListSink(), ListSink()
    tracer = Tracer(sinks=[sink_a, sink_b])

    with tracer.span("op"):
        pass

    assert len(sink_a.spans) == 1
    assert len(sink_b.spans) == 1
