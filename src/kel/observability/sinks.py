"""Sinks receive completed Spans. This is the swap point between console
output (default, zero-dependency) and real backends like Grafana/Tempo
(via `kel.observability.otel.OTelSink`, `pip install kel[otel]`)."""

from __future__ import annotations

from typing import Protocol

from kel.observability.types import Span


class Sink(Protocol):
    def emit(self, span: Span) -> None: ...


class ConsoleSink:
    """Zero-dependency default: one line per span to stdout."""

    def emit(self, span: Span) -> None:
        status = "ERR" if span.status == "error" else "ok"
        attrs = " ".join(f"{k}={v}" for k, v in span.attributes.items())
        line = f"[kel] {span.name} {status} {span.duration_ms:.0f}ms {attrs}".rstrip()
        if span.error:
            line += f" error={span.error!r}"
        print(line)


class ListSink:
    """Collects spans in memory. Used by tests and by kel.testing's replay
    tooling (DESIGN.md 3.7) to assert on trace shape."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def emit(self, span: Span) -> None:
        self.spans.append(span)


class NullSink:
    """Discards everything. Useful for benchmarks or opting out of tracing overhead."""

    def emit(self, span: Span) -> None:
        pass
