"""Aggregates spans into monitoring metrics. It's just another
`kel.observability.Sink` — plug it in with `add_sink(metrics)` the same
way you'd add any other sink, no separate mechanism to learn.

Stdlib only (threading, collections) — this is deliberately the
"lightweight" tier: in-memory, per-process, resets on restart. For
durable/cross-process metrics, export spans to Grafana via `kel[otel]`
instead; this is for a quick local/dev-facing view, not a production
metrics backend.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any

from kel.observability.types import Span


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


class MetricsSink:
    def __init__(self, *, max_recent: int = 200, max_latency_samples: int = 1000):
        self._lock = threading.Lock()
        self._max_latency_samples = max_latency_samples
        self.total_calls = 0
        self.error_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._by_name: dict[str, int] = defaultdict(int)
        self._errors_by_name: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_latency_samples))
        self._recent: deque[Span] = deque(maxlen=max_recent)

    def emit(self, span: Span) -> None:
        with self._lock:
            self.total_calls += 1
            self._by_name[span.name] += 1
            self._latencies[span.name].append(span.duration_ms)
            if span.status == "error":
                self.error_count += 1
                self._errors_by_name[span.name] += 1

            attrs = span.attributes
            if "input_tokens" in attrs:
                self.total_input_tokens += attrs["input_tokens"]
            if "output_tokens" in attrs:
                self.total_output_tokens += attrs["output_tokens"]
            if span.name == "kel.model.cache":
                if attrs.get("hit"):
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1

            self._recent.append(span)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            all_latencies = sorted(v for values in self._latencies.values() for v in values)
            by_name = {}
            for name, count in self._by_name.items():
                latencies = sorted(self._latencies[name])
                by_name[name] = {
                    "calls": count,
                    "errors": self._errors_by_name.get(name, 0),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                    "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
                }
            cache_total = self.cache_hits + self.cache_misses
            recent = [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_ms": round(s.duration_ms, 1),
                    "start_time": s.start_time,
                    "attributes": s.attributes,
                    "error": s.error,
                }
                for s in reversed(self._recent)
            ]
            return {
                "total_calls": self.total_calls,
                "error_count": self.error_count,
                "error_rate": round(self.error_count / self.total_calls, 4) if self.total_calls else 0.0,
                "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0.0,
                "p95_latency_ms": round(_percentile(all_latencies, 0.95), 1),
                "p99_latency_ms": round(_percentile(all_latencies, 0.99), 1),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hits / cache_total, 4) if cache_total else 0.0,
                "by_name": by_name,
                "recent": recent,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_calls = 0
            self.error_count = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self._by_name.clear()
            self._errors_by_name.clear()
            self._latencies.clear()
            self._recent.clear()
