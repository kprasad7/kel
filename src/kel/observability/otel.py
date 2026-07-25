"""Bridge from kel's internal Span/Sink to real OpenTelemetry, for export
to Grafana Tempo (traces) via an OTel Collector. Requires `pip install
kel[otel]` — kept out of core so console tracing has zero dependencies."""

from __future__ import annotations


class OTelSink:
    def __init__(self, tracer_name: str = "kel"):
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise ImportError(
                "OTelSink requires the opentelemetry SDK. Install it with `pip install kel[otel]`."
            ) from exc
        self._tracer = trace.get_tracer(tracer_name)

    def emit(self, span) -> None:
        from opentelemetry.trace import Status, StatusCode

        start_ns = int(span.start_time * 1_000_000_000)
        end_ns = start_ns + int(span.duration_ms * 1_000_000)
        otel_span = self._tracer.start_span(span.name, start_time=start_ns)
        for key, value in span.attributes.items():
            otel_span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
        if span.status == "error":
            otel_span.set_status(Status(StatusCode.ERROR, span.error))
        otel_span.end(end_time=end_ns)


def configure_otlp(endpoint: str | None = None, *, service_name: str = "kel") -> None:
    """Set up a real OTel TracerProvider exporting via OTLP (e.g. to an
    OpenTelemetry Collector feeding Grafana Tempo/Prometheus/Loki), and
    attach an OTelSink to kel's default tracer so every kel span exports.

    `endpoint` defaults to the standard OTEL_EXPORTER_OTLP_ENDPOINT env var
    / OTel SDK default (http://localhost:4317) when not given.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ImportError(
            "configure_otlp requires the opentelemetry SDK + OTLP exporter. "
            "Install with `pip install kel[otel]`."
        ) from exc

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from kel.observability.tracer import add_sink

    add_sink(OTelSink())
