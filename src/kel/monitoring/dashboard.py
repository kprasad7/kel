"""Lightweight monitoring dashboard: one self-contained HTML page (inline
CSS/JS, no external assets, no CDN) that polls a JSON endpoint every 1.5s
for a live-updating view of `MetricsSink`'s aggregated stats and a
scrolling recent-activity log — the "live refresh logs" piece.

Stdlib-only (`http.server`), same pattern as `kel.sdk.serve` — this is
explicitly not trying to be Grafana. For durable/production dashboards,
export to Grafana via `kel[otel]` instead; this is a zero-dependency
local/dev view you can point a browser at in one line.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from kel.monitoring.metrics import MetricsSink

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>kel monitoring</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; padding: 24px; background: #0d1117; color: #c9d1d9; }
h1 { font-size: 18px; margin: 0 0 16px; color: #58a6ff; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; min-width: 120px; }
.card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 500; }
.log { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; max-height: 320px; overflow-y: auto; font-family: monospace; font-size: 12px; }
.log div { padding: 3px 4px; border-bottom: 1px solid #21262d; }
.ok { color: #3fb950; } .error { color: #f85149; }
.muted { color: #8b949e; }
</style></head>
<body>
<h1>kel monitoring</h1>
<div class="cards" id="cards"></div>
<table id="byname"><thead><tr><th>span</th><th>calls</th><th>errors</th><th>avg ms</th><th>p95 ms</th></tr></thead><tbody></tbody></table>
<div class="log" id="log"></div>
<script>
function card(label, value) {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}
async function refresh() {
  const resp = await fetch("/metrics.json");
  const m = await resp.json();
  document.getElementById("cards").innerHTML = [
    card("Total calls", m.total_calls),
    card("Error rate", (m.error_rate * 100).toFixed(1) + "%"),
    card("Avg latency", m.avg_latency_ms + " ms"),
    card("p95 latency", m.p95_latency_ms + " ms"),
    card("Input tokens", m.total_input_tokens),
    card("Output tokens", m.total_output_tokens),
    card("Cache hit rate", (m.cache_hit_rate * 100).toFixed(1) + "%"),
  ].join("");

  const rows = Object.entries(m.by_name).map(([name, s]) =>
    `<tr><td>${name}</td><td>${s.calls}</td><td>${s.errors}</td><td>${s.avg_latency_ms}</td><td>${s.p95_latency_ms}</td></tr>`
  ).join("");
  document.querySelector("#byname tbody").innerHTML = rows;

  const log = m.recent.map(s => {
    const cls = s.status === "error" ? "error" : "ok";
    const time = new Date(s.start_time * 1000).toLocaleTimeString();
    return `<div><span class="muted">${time}</span> <span class="${cls}">${s.status}</span> ${s.name} — ${s.duration_ms}ms</div>`;
  }).join("");
  document.getElementById("log").innerHTML = log;
}
refresh();
setInterval(refresh, 1500);
</script>
</body></html>"""


def _make_handler(metrics: MetricsSink) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._respond(200, _PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
            elif self.path == "/metrics.json":
                data = json.dumps(metrics.snapshot()).encode("utf-8")
                self._respond(200, data, content_type="application/json")
            else:
                self._respond(404, b"not found", content_type="text/plain")

        def _respond(self, status: int, data: bytes, *, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass

    return Handler


class MonitoringDashboard:
    def __init__(self, metrics: MetricsSink, *, host: str = "127.0.0.1", port: int = 0):
        self.metrics = metrics
        self._httpd = HTTPServer((host, port), _make_handler(metrics))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> "MonitoringDashboard":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_dashboard(metrics: MetricsSink, *, host: str = "127.0.0.1", port: int = 0) -> MonitoringDashboard:
    dashboard = MonitoringDashboard(metrics, host=host, port=port)
    dashboard.start()
    return dashboard
