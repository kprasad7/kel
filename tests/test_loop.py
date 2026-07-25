import pytest

from kel.context import Loop, LoopBudgetExceededError, StuckLoopError


def test_step_increments_and_allows_up_to_max_iterations():
    loop = Loop(max_iterations=3)
    assert loop.step() == 1
    assert loop.step() == 2
    assert loop.step() == 3


def test_step_raises_once_max_iterations_exceeded():
    loop = Loop(max_iterations=2)
    loop.step()
    loop.step()
    with pytest.raises(LoopBudgetExceededError):
        loop.step()


def test_record_action_raises_when_action_repeats_past_threshold():
    loop = Loop(max_iterations=10, stuck_window=4, stuck_threshold=3)
    loop.step()
    loop.record_action("search:foo")
    loop.step()
    loop.record_action("search:foo")
    loop.step()
    with pytest.raises(StuckLoopError) as exc_info:
        loop.record_action("search:foo")
    assert exc_info.value.signature == "search:foo"
    assert exc_info.value.repeat_count == 3


def test_record_action_does_not_raise_for_varied_actions():
    loop = Loop(max_iterations=10, stuck_window=4, stuck_threshold=3)
    for sig in ["a", "b", "c", "d", "a", "b"]:
        loop.step()
        loop.record_action(sig)  # never 3 repeats within the sliding window of 4


def test_exhausted_reflects_iteration_vs_max():
    loop = Loop(max_iterations=2)
    assert not loop.exhausted
    loop.step()
    assert not loop.exhausted
    loop.step()
    assert loop.exhausted
