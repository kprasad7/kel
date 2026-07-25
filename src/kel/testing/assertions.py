"""Assert on trace *shape* (DESIGN.md 3.7: "assert on trace shape — which
tools called, in what order — not just final output") plus a couple of
budget/loop invariant checks. v1's invariant checking is plain assertion
helpers over already-executed runs, not real property-based fuzzing
(hypothesis-style generated-input testing) — that's a reasonable future
upgrade, not something claimed here."""

from __future__ import annotations

from kel.budget.tracker import BudgetTracker
from kel.observability.types import Span


def assert_node_sequence(history: list[str], expected: list[str]) -> None:
    if history != expected:
        raise AssertionError(f"node sequence mismatch: expected {expected}, got {history}")


def assert_nodes_visited(history: list[str], expected_subset: set[str]) -> None:
    missing = expected_subset - set(history)
    if missing:
        raise AssertionError(f"expected node(s) not visited: {sorted(missing)}")


def assert_span_names(spans: list[Span], expected: list[str]) -> None:
    actual = [s.name for s in spans]
    if actual != expected:
        raise AssertionError(f"span sequence mismatch: expected {expected}, got {actual}")


def assert_no_error_spans(spans: list[Span]) -> None:
    errors = [s for s in spans if s.status == "error"]
    if errors:
        raise AssertionError(f"found {len(errors)} error span(s): {[s.name for s in errors]}")


def assert_budget_never_exceeded(tracker: BudgetTracker) -> None:
    snapshot = tracker.snapshot()
    if snapshot.tokens_remaining is not None and snapshot.tokens_remaining < 0:
        raise AssertionError(f"token budget exceeded by {-snapshot.tokens_remaining}")
    if snapshot.cost_usd_remaining is not None and snapshot.cost_usd_remaining < 0:
        raise AssertionError(f"cost budget exceeded by ${-snapshot.cost_usd_remaining:.4f}")
    if snapshot.tool_calls_remaining is not None and snapshot.tool_calls_remaining < 0:
        raise AssertionError(f"tool call budget exceeded by {-snapshot.tool_calls_remaining}")
