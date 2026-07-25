import threading
import time

import pytest

from kel.context import Loop, LoopBudgetExceededError, StuckLoopError
from kel.runtime import END, Graph, InMemoryCheckpointStore, run_graph


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
