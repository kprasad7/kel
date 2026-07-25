import pytest

from kel.runtime import Graph, InMemoryCheckpointStore, Interrupt, resume_graph, run_graph


def test_node_raising_interrupt_pauses_the_run():
    graph = Graph(entry="approve")

    def approve(state):
        if "__resume_value__" not in state:
            raise Interrupt({"action": "approve_payment", "amount": 100})
        return {"approved": state["__resume_value__"]}

    graph.add_node("approve", approve)
    graph.set_finish("approve")

    result = run_graph(graph, {})

    assert result.interrupted is True
    assert result.interrupt_payload == {"action": "approve_payment", "amount": 100}
    assert result.pending_node == "approve"
    assert "approved" not in result.state


def test_resume_graph_continues_from_pending_node_with_resume_value():
    graph = Graph(entry="approve")

    def approve(state):
        if "__resume_value__" not in state:
            raise Interrupt({"action": "approve_payment", "amount": 100})
        return {"approved": state["__resume_value__"]}

    graph.add_node("approve", approve)
    graph.set_finish("approve")

    paused = run_graph(graph, {})
    resumed = resume_graph(graph, paused, resume_value=True)

    assert resumed.interrupted is False
    assert resumed.state["approved"] is True
    assert "approve" in resumed.history


def test_resume_graph_raises_if_run_was_not_interrupted():
    graph = Graph(entry="a")
    graph.add_node("a", lambda state: {})
    graph.set_finish("a")

    completed = run_graph(graph, {})

    with pytest.raises(ValueError):
        resume_graph(graph, completed, resume_value=None)


def test_interrupt_mid_pipeline_preserves_earlier_history_and_state():
    graph = Graph(entry="a")

    def a(state):
        return {"a_ran": True}

    def b(state):
        if "__resume_value__" not in state:
            raise Interrupt("need human input for b")
        return {"b_ran": True, "human_said": state["__resume_value__"]}

    def c(state):
        return {"c_ran": True}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_node("c", c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_finish("c")

    paused = run_graph(graph, {})
    assert paused.interrupted
    assert paused.pending_node == "b"
    assert paused.state == {"a_ran": True}  # b's update never applied
    assert paused.history == ["a"]

    resumed = resume_graph(graph, paused, resume_value="yes")

    assert resumed.state["a_ran"] is True
    assert resumed.state["b_ran"] is True
    assert resumed.state["human_said"] == "yes"
    assert resumed.state["c_ran"] is True
    assert resumed.history == ["a", "b", "c"]


def test_interrupt_with_checkpoint_store_still_checkpoints_completed_nodes():
    graph = Graph(entry="a")

    def a(state):
        return {"a_ran": True}

    def b(state):
        raise Interrupt("stop here")

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge("a", "b")
    graph.set_finish("b")

    store = InMemoryCheckpointStore()
    paused = run_graph(graph, {}, checkpoint_store=store, run_id="run-1")

    assert paused.interrupted
    history = store.history("run-1")
    assert [c.node for c in history] == ["a"]  # b never completed, never checkpointed
