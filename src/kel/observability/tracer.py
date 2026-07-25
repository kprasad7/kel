"""Span tracer. Deliberately not built on the real `opentelemetry` SDK
directly — that's a heavy hard dependency for "observability is not
optional" to require of every kel install. Instead kel has its own minimal
Span type and Sink protocol; `kel.observability.otel.OTelSink` bridges to
real OTel/Grafana for anyone who installs `kel[otel]`.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from kel.observability.sinks import ConsoleSink, Sink
from kel.observability.types import Span

_current_span_id: ContextVar[str | None] = ContextVar("kel_current_span_id", default=None)


class SpanHandle:
    """Live handle yielded by `Tracer.span()`. Call `set_attribute` while
    the span is open; it's converted to an immutable `Span` on exit."""

    def __init__(self, name: str, parent_id: str | None, attributes: dict[str, Any]):
        self.id = uuid.uuid4().hex
        self.name = name
        self.parent_id = parent_id
        self.attributes = attributes
        self._start_perf = time.perf_counter()
        self._start_wall = time.time()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def _finish(self, status: Literal["ok", "error"], error: str | None) -> Span:
        duration_ms = (time.perf_counter() - self._start_perf) * 1000
        return Span(
            id=self.id,
            parent_id=self.parent_id,
            name=self.name,
            start_time=self._start_wall,
            duration_ms=duration_ms,
            attributes=dict(self.attributes),
            status=status,
            error=error,
        )


class Tracer:
    def __init__(self, sinks: list[Sink] | None = None):
        self.sinks: list[Sink] = sinks if sinks is not None else [ConsoleSink()]

    def add_sink(self, sink: Sink) -> None:
        self.sinks.append(sink)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanHandle]:
        parent_id = _current_span_id.get()
        handle = SpanHandle(name, parent_id, dict(attributes))
        token = _current_span_id.set(handle.id)
        status: Literal["ok", "error"] = "ok"
        error: str | None = None
        try:
            yield handle
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            _current_span_id.reset(token)
            finished = handle._finish(status, error)
            for sink in self.sinks:
                sink.emit(finished)


_default_tracer = Tracer()


def get_tracer() -> Tracer:
    return _default_tracer


def configure(sinks: list[Sink]) -> None:
    """Replace the default tracer's sinks entirely."""
    _default_tracer.sinks = sinks


def add_sink(sink: Sink) -> None:
    """Add a sink (e.g. OTelSink) alongside whatever's already configured."""
    _default_tracer.add_sink(sink)
