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
    # if slow ran sequentially after fast, total time would be >= slow's
    # sleep duration measured from *before* fast starts; if concurrent,
    # slow's timer starts before fast() returns.
    start = time.monotonic()

    def fast():
        time.sleep(0.05)
        return "filler"

    def slow():
        time.sleep(0.15)
        return "final"

    result = run_dual_path(fast, slow)
    elapsed = time.monotonic() - start

    assert result == "final"
    # sequential would take >= 0.05 + 0.15 = 0.20s; concurrent should be close to max(0.05, 0.15) = 0.15s
    assert elapsed < 0.19


def test_dual_path_with_no_filler_still_returns_final():
    result = run_dual_path(lambda: None, lambda: "final answer")
    assert result == "final answer"


def test_dual_path_without_on_filler_callback_does_not_error():
    result = run_dual_path(lambda: "filler text", lambda: "final")
    assert result == "final"
