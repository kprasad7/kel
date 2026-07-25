import json
import urllib.error
import urllib.request

from kel.monitoring import MetricsSink, start_dashboard
from kel.observability.types import Span


def _span(name, *, duration_ms=10.0, status="ok", attributes=None, error=None, start_time=0.0):
    return Span(
        id="s1", name=name, start_time=start_time, duration_ms=duration_ms,
        attributes=attributes or {}, status=status, error=error,
    )


def test_metrics_sink_counts_calls_and_errors():
    metrics = MetricsSink()
    metrics.emit(_span("kel.model.generate"))
    metrics.emit(_span("kel.model.generate", status="error", error="boom"))

    snap = metrics.snapshot()
    assert snap["total_calls"] == 2
    assert snap["error_count"] == 1
    assert snap["error_rate"] == 0.5


def test_metrics_sink_tracks_latency_and_percentiles():
    metrics = MetricsSink()
    for ms in [10, 20, 30, 40, 100]:
        metrics.emit(_span("kel.model.generate", duration_ms=ms))

    snap = metrics.snapshot()
    assert snap["avg_latency_ms"] == 40.0
    assert snap["p95_latency_ms"] == 100.0


def test_metrics_sink_sums_token_usage():
    metrics = MetricsSink()
    metrics.emit(_span("kel.model.generate", attributes={"input_tokens": 100, "output_tokens": 50}))
    metrics.emit(_span("kel.model.generate", attributes={"input_tokens": 200, "output_tokens": 75}))

    snap = metrics.snapshot()
    assert snap["total_input_tokens"] == 300
    assert snap["total_output_tokens"] == 125


def test_metrics_sink_tracks_cache_hit_rate():
    metrics = MetricsSink()
    metrics.emit(_span("kel.model.cache", attributes={"hit": True}))
    metrics.emit(_span("kel.model.cache", attributes={"hit": True}))
    metrics.emit(_span("kel.model.cache", attributes={"hit": False}))

    snap = metrics.snapshot()
    assert snap["cache_hits"] == 2
    assert snap["cache_misses"] == 1
    assert snap["cache_hit_rate"] == round(2 / 3, 4)


def test_metrics_sink_breaks_down_by_span_name():
    metrics = MetricsSink()
    metrics.emit(_span("kel.model.generate", duration_ms=10))
    metrics.emit(_span("kel.runtime.node", duration_ms=20))
    metrics.emit(_span("kel.runtime.node", duration_ms=30, status="error"))

    snap = metrics.snapshot()
    assert snap["by_name"]["kel.model.generate"]["calls"] == 1
    assert snap["by_name"]["kel.runtime.node"]["calls"] == 2
    assert snap["by_name"]["kel.runtime.node"]["errors"] == 1


def test_metrics_sink_recent_log_is_most_recent_first():
    metrics = MetricsSink()
    metrics.emit(_span("first"))
    metrics.emit(_span("second"))
    metrics.emit(_span("third"))

    snap = metrics.snapshot()
    assert [r["name"] for r in snap["recent"]] == ["third", "second", "first"]


def test_metrics_sink_recent_log_is_bounded():
    metrics = MetricsSink(max_recent=3)
    for i in range(10):
        metrics.emit(_span(f"span-{i}"))

    snap = metrics.snapshot()
    assert len(snap["recent"]) == 3
    assert snap["recent"][0]["name"] == "span-9"


def test_metrics_sink_reset_clears_everything():
    metrics = MetricsSink()
    metrics.emit(_span("a"))
    metrics.reset()

    snap = metrics.snapshot()
    assert snap["total_calls"] == 0
    assert snap["recent"] == []


def test_metrics_sink_empty_snapshot_has_no_errors():
    metrics = MetricsSink()
    snap = metrics.snapshot()
    assert snap["total_calls"] == 0
    assert snap["error_rate"] == 0.0
    assert snap["cache_hit_rate"] == 0.0


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type")


def test_dashboard_serves_html_page():
    metrics = MetricsSink()
    with start_dashboard(metrics, port=0) as dashboard:
        status, body, content_type = _get(dashboard.url)
        assert status == 200
        assert "text/html" in content_type
        assert b"kel monitoring" in body


def test_dashboard_serves_live_metrics_json():
    metrics = MetricsSink()
    metrics.emit(_span("kel.model.generate", attributes={"input_tokens": 5, "output_tokens": 3}))

    with start_dashboard(metrics, port=0) as dashboard:
        status, body, content_type = _get(dashboard.url + "metrics.json")
        data = json.loads(body)

        assert status == 200
        assert "application/json" in content_type
        assert data["total_calls"] == 1
        assert data["total_input_tokens"] == 5


def test_dashboard_metrics_reflect_new_spans_without_restart():
    # proves the "live refresh" contract: metrics.json reflects spans
    # emitted *after* the dashboard started, no restart needed
    metrics = MetricsSink()
    with start_dashboard(metrics, port=0) as dashboard:
        _, body1, _ = _get(dashboard.url + "metrics.json")
        assert json.loads(body1)["total_calls"] == 0

        metrics.emit(_span("kel.model.generate"))

        _, body2, _ = _get(dashboard.url + "metrics.json")
        assert json.loads(body2)["total_calls"] == 1


def test_dashboard_unknown_path_returns_404():
    metrics = MetricsSink()
    with start_dashboard(metrics, port=0) as dashboard:
        try:
            _get(dashboard.url + "nope")
            assert False, "expected HTTPError"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
