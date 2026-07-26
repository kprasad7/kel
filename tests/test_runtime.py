import threading

import pytest

from kel.context import Loop, LoopBudgetExceededError, StuckLoopError
from kel.runtime import END, Graph, InMemoryCheckpointStore, fork_from_checkpoint, run_graph


def test_linear_graph_runs_nodes_in_order():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"a_ran": True})
    graph.add_node("b", lambda state: {"b_ran": True})
    graph.add_edge("a", "b")
    graph.set_finish("b")

    result = run_graph(graph, {})

    assert result.history == ["a", "b"]
    assert result.state == {"a_ran": True, "b_ran": True}


def test_conditional_edge_branches_on_state():
    graph = Graph(entry="check")
    graph.add_node("check", lambda state: {"seen": True})
    graph.add_node("high", lambda state: {"branch": "high"})
    graph.add_node("low", lambda state: {"branch": "low"})
    graph.add_conditional_edges("check", lambda state: "high" if state.get("value", 0) > 5 else "low")
    graph.set_finish("high")
    graph.set_finish("low")

    result = run_graph(graph, {"value": 10})
    assert result.state["branch"] == "high"

    result2 = run_graph(graph, {"value": 1})
    assert result2.state["branch"] == "low"


def test_cyclic_graph_loops_until_condition_met():
    graph = Graph(entry="counter")

    def increment(state):
        return {"count": state.get("count", 0) + 1}

    graph.add_node("counter", increment)
    graph.add_conditional_edges("counter", lambda state: END if state["count"] >= 5 else "counter")

    result = run_graph(graph, {})

    assert result.state["count"] == 5
    assert result.history == ["counter"] * 5


def test_cyclic_graph_revisiting_same_node_is_not_treated_as_stuck():
    # revisiting "counter" 10 times is normal agentic-loop behavior; it
    # must NOT raise StuckLoopError just because the node name repeats.
    graph = Graph(entry="counter")
    graph.add_node("counter", lambda state: {"count": state.get("count", 0) + 1})
    graph.add_conditional_edges("counter", lambda state: END if state["count"] >= 10 else "counter")

    result = run_graph(graph, {}, loop=Loop(max_iterations=20, stuck_window=4, stuck_threshold=3))
    assert result.state["count"] == 10


def test_node_can_opt_into_stuck_loop_detection_via_signature():
    graph = Graph(entry="agent")
    graph.add_node("agent", lambda state: {"__signature__": "search:same-args"})
    graph.add_conditional_edges("agent", lambda state: "agent")  # never terminates on its own

    with pytest.raises(StuckLoopError):
        run_graph(graph, {}, loop=Loop(max_iterations=20, stuck_window=4, stuck_threshold=3))


def test_loop_budget_exceeded_propagates():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {})
    graph.add_edge("a", "a")

    with pytest.raises(LoopBudgetExceededError):
        run_graph(graph, {}, loop=Loop(max_iterations=3))


def test_parallel_fan_out_runs_layer_nodes_concurrently():
    barrier = threading.Barrier(2, timeout=2)

    def slow_node(state):
        barrier.wait()  # only succeeds if both nodes are running at the same time
        return {}

    graph = Graph(entry="start")
    graph.add_node("start", lambda state: {})
    graph.add_node("x", slow_node)
    graph.add_node("y", slow_node)
    graph.add_conditional_edges("start", lambda state: ["x", "y"])
    graph.set_finish("x")
    graph.set_finish("y")

    result = run_graph(graph, {})
    assert set(result.history) == {"start", "x", "y"}


def test_fan_out_results_merge_deterministically_by_layer_order():
    graph = Graph(entry="start")
    graph.add_node("start", lambda state: {})
    graph.add_node("x", lambda state: {"winner": "x"})
    graph.add_node("y", lambda state: {"winner": "y"})
    graph.add_conditional_edges("start", lambda state: ["x", "y"])
    graph.set_finish("x")
    graph.set_finish("y")

    result = run_graph(graph, {})
    # "y" runs after "x" in the layer, so its update wins the merge
    assert result.state["winner"] == "y"


def test_checkpoint_store_records_every_node_transition():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"step": "a"})
    graph.add_node("b", lambda state: {"step": "b"})
    graph.add_edge("a", "b")
    graph.set_finish("b")

    store = InMemoryCheckpointStore()
    result = run_graph(graph, {}, checkpoint_store=store, run_id="run-1")

    history = store.history("run-1")
    assert [c.node for c in history] == ["a", "b"]
    assert history[-1].state == result.state
    assert store.load_latest("run-1").node == "b"


def test_fork_from_checkpoint_rewinds_and_continues_from_an_arbitrary_step():
    # a 3-step linear graph: a -> b -> c, where "b" is the step we'll
    # rewind to and re-run from, with a patched state variable.
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"count": 1})
    graph.add_node("b", lambda state: {"count": state["count"] + 1})
    graph.add_node("c", lambda state: {"final": state["count"] * 10})
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_finish("c")

    store = InMemoryCheckpointStore()
    original = run_graph(graph, {}, checkpoint_store=store, run_id="run-1")
    assert original.state["final"] == 20  # count: 1 (a) -> 2 (b) -> final = 2*10 (c)

    checkpoint_at_b = next(c for c in store.history("run-1") if c.node == "b")

    forked = fork_from_checkpoint(graph, checkpoint_at_b, state_overrides={"count": 100})

    # forked continues from after "b" (i.e. runs "c") using the overridden count,
    # without re-running "a" or "b"
    assert forked.history == ["b", "c"]
    assert forked.state["final"] == 1000
    assert forked.run_id != "run-1"  # a new timeline, not a continuation of the original


def test_fork_from_checkpoint_uses_checkpoint_state_when_no_override_given():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"count": 1})
    graph.add_node("b", lambda state: {"count": state["count"] + 1})
    graph.add_edge("a", "b")
    graph.set_finish("b")

    store = InMemoryCheckpointStore()
    run_graph(graph, {}, checkpoint_store=store, run_id="run-1")
    checkpoint_at_a = next(c for c in store.history("run-1") if c.node == "a")

    forked = fork_from_checkpoint(graph, checkpoint_at_a)

    assert forked.state["count"] == 2  # re-derived from checkpoint's own state, unmodified


def test_fork_from_checkpoint_can_write_to_a_new_checkpoint_store():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {"count": 1})
    graph.add_node("b", lambda state: {"count": state["count"] + 1})
    graph.add_edge("a", "b")
    graph.set_finish("b")

    original_store = InMemoryCheckpointStore()
    run_graph(graph, {}, checkpoint_store=original_store, run_id="run-1")
    checkpoint_at_a = next(c for c in original_store.history("run-1") if c.node == "a")

    fork_store = InMemoryCheckpointStore()
    forked = fork_from_checkpoint(graph, checkpoint_at_a, checkpoint_store=fork_store)

    assert fork_store.history(forked.run_id)  # the forked timeline got its own checkpoints
    assert original_store.history("run-1")  # the original timeline is untouched


def test_node_failure_without_a_fallback_still_raises():
    # baseline: a plain node crash crashes the run, same as before —
    # fallback routing is opt-in via set_fallback, not automatic swallowing
    graph = Graph(entry="risky")

    def risky(state):
        raise ValueError("boom")

    graph.add_node("risky", risky)
    graph.set_finish("risky")

    with pytest.raises(ValueError, match="boom"):
        run_graph(graph, {})


def test_node_failure_with_a_fallback_routes_there_instead_of_crashing():
    graph = Graph(entry="risky")

    def risky(state):
        raise ValueError("boom")

    graph.add_node("risky", risky)
    graph.add_node("recover", lambda state: {"recovered": True, "saw_error": state["__error__"]["error"]})
    graph.set_fallback("risky", "recover")
    graph.set_finish("recover")

    result = run_graph(graph, {})

    assert result.state["recovered"] is True
    assert result.state["saw_error"] == "boom"
    assert result.state["__error__"] == {"node": "risky", "error": "boom"}
    assert result.history == ["risky", "recover"]


def test_fallback_node_can_itself_continue_the_graph_normally():
    def risky(state):
        raise ValueError("boom")

    graph = Graph(entry="risky")
    graph.add_node("risky", risky)
    graph.add_node("recover", lambda state: {"recovered": True})
    graph.add_node("finish", lambda state: {"done": True})
    graph.set_fallback("risky", "recover")
    graph.add_edge("recover", "finish")
    graph.set_finish("finish")

    result = run_graph(graph, {})

    assert result.history == ["risky", "recover", "finish"]
    assert result.state == {"recovered": True, "done": True, "__error__": {"node": "risky", "error": "boom"}}


def test_checkpoint_store_still_records_the_failed_node_before_the_fallback():
    graph = Graph(entry="risky")

    def risky(state):
        raise ValueError("boom")

    graph.add_node("risky", risky)
    graph.add_node("recover", lambda state: {"recovered": True})
    graph.set_fallback("risky", "recover")
    graph.set_finish("recover")

    store = InMemoryCheckpointStore()
    run_graph(graph, {}, checkpoint_store=store, run_id="run-1")

    history = store.history("run-1")
    assert [c.node for c in history] == ["risky", "recover"]
    assert history[0].state["__error__"]["error"] == "boom"


def test_dedupes_duplicate_next_nodes_within_a_layer():
    calls = []

    def track(name):
        def fn(state):
            calls.append(name)
            return {}
        return fn

    graph = Graph(entry="a")
    graph.add_node("a", track("a"))
    graph.add_node("b", track("b"))
    graph.add_conditional_edges("a", lambda state: ["b", "b"])  # two branches both requesting "b"
    graph.set_finish("b")

    run_graph(graph, {})
    assert calls.count("b") == 1


def test_validate_raises_for_missing_entry_node():
    graph = Graph(entry="missing")
    with pytest.raises(ValueError):
        run_graph(graph, {})
