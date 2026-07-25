import time

from kel.realtime import run_dual_path


def test_dual_path_emits_filler_before_slow_path_finishes():
    events = []

    def fast():
        events.append(("fast", time.monotonic()))
        return "let me check that"

    def slow():
        time.sleep(0.1)
        events.append(("slow", time.monotonic()))
        return "the final answer"

    fillers = []
    final = run_dual_path(fast, slow, on_filler=fillers.append)

    assert final == "the final answer"
    assert fillers == ["let me check that"]
    assert events[0][0] == "fast"
    assert events[1][0] == "slow"


def test_dual_path_runs_slow_concurrently_not_after_fast():
    # Assert concurrency via event ordering, not a wall-clock threshold —
    # tight sleep-based timing budgets are flaky on loaded/slow CI runners
    # (observed on macOS runners: 0.30s actual vs a 0.19s ceiling, despite
    # correct concurrent behavior). Recording each side's start time proves
    # `slow` started before `fast` returned, regardless of how long either
    # actually takes on a given machine.
    slow_started_at = None
    fast_returned_at = None

    def fast():
        nonlocal fast_returned_at
        time.sleep(0.05)
        fast_returned_at = time.monotonic()
        return "filler"

    def slow():
        nonlocal slow_started_at
        slow_started_at = time.monotonic()
        time.sleep(0.15)
        return "final"

    result = run_dual_path(fast, slow)

    assert result == "final"
    assert slow_started_at is not None
    assert fast_returned_at is not None
    assert slow_started_at < fast_returned_at


def test_dual_path_with_no_filler_still_returns_final():
    result = run_dual_path(lambda: None, lambda: "final answer")
    assert result == "final answer"


def test_dual_path_without_on_filler_callback_does_not_error():
    result = run_dual_path(lambda: "filler text", lambda: "final")
    assert result == "final"
