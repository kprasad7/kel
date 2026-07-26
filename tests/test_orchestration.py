from helpers import ScriptedModel
from kel.agents import Agent, agent_node, run_parallel, run_supervisor, run_swarm, sequential_pipeline
from kel.models.types import ModelResponse, TextPart, Usage
from kel.runtime import END, Graph, run_graph


def _text_response(text: str, rid: str = "r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())


def test_sequential_pipeline_passes_upstream_output_downstream():
    model_a = ScriptedModel("fake-1", [_text_response("A's answer", "a1")])
    agent_a = Agent("agent_a", model_a)

    model_b = ScriptedModel("fake-1", [_text_response("B's answer", "b1")])
    agent_b = Agent("agent_b", model_b)

    graph = sequential_pipeline([agent_a, agent_b])
    result = run_graph(graph, {"input": "do the task"})

    assert result.state["agent_a_output"] == "A's answer"
    assert result.state["agent_b_output"] == "B's answer"
    seen_messages = model_b.calls[0][0]
    assert any("A's answer" in m.text for m in seen_messages)


def test_sequential_pipeline_context_selector_scopes_what_each_agent_sees():
    model_a = ScriptedModel("fake-1", [_text_response("A's answer", "a1")])
    agent_a = Agent("agent_a", model_a)
    model_b = ScriptedModel("fake-1", [_text_response("B's answer", "b1")])
    agent_b = Agent("agent_b", model_b)
    model_c = ScriptedModel("fake-1", [_text_response("C's answer", "c1")])
    agent_c = Agent("agent_c", model_c)

    # agent_c should only see agent_a's output, not agent_b's
    def only_from_a(agent_name, state):
        return {k: v for k, v in state.items() if k.startswith("agent_a_") or k == "input"}

    graph = sequential_pipeline([agent_a, agent_b, agent_c], context_selector=only_from_a)
    run_graph(graph, {"input": "do the task"})

    seen_messages = model_c.calls[0][0]
    combined_text = " ".join(m.text for m in seen_messages)
    assert "A's answer" in combined_text
    assert "B's answer" not in combined_text


def test_run_supervisor_results_selector_scopes_what_supervisor_sees():
    supervisor_model = ScriptedModel(
        "fake-1",
        [
            _text_response("DELEGATE: worker1 :: do subtask 1", "s1"),
            _text_response("DELEGATE: worker2 :: do subtask 2", "s2"),
            _text_response("DONE: all done", "s3"),
        ],
    )
    supervisor = Agent("supervisor", supervisor_model)
    worker1 = Agent("worker1", ScriptedModel("fake-1", [_text_response("result 1", "w1")]))
    worker2 = Agent("worker2", ScriptedModel("fake-1", [_text_response("result 2", "w2")]))

    result = run_supervisor(
        supervisor,
        {"worker1": worker1, "worker2": worker2},
        "big task",
        max_rounds=5,
        results_selector=lambda results: {},  # show the supervisor nothing
    )

    assert result["final_answer"] == "all done"
    # the actual results are still recorded, just not shown to the supervisor
    assert result["results"] == {"worker1": "result 1", "worker2": "result 2"}
    prompts = [messages[-1].text for messages, _ in supervisor_model.calls]
    assert all("result 1" not in p and "result 2" not in p for p in prompts)


def test_run_parallel_fans_out_and_merges():
    model_a = ScriptedModel("fake-1", [_text_response("from A", "a")])
    model_b = ScriptedModel("fake-1", [_text_response("from B", "b")])
    agents = {"a": Agent("a", model_a), "b": Agent("b", model_b)}

    result = run_parallel(agents, "task")

    assert result["results"] == {"a": "from A", "b": "from B"}
    assert "from A" in result["merged"]
    assert "from B" in result["merged"]


def test_run_parallel_uses_custom_merge_function():
    model_a = ScriptedModel("fake-1", [_text_response("x", "a")])
    model_b = ScriptedModel("fake-1", [_text_response("y", "b")])
    agents = {"a": Agent("a", model_a), "b": Agent("b", model_b)}

    result = run_parallel(agents, "task", merge=lambda results: "|".join(sorted(results.values())))

    assert result["merged"] == "x|y"


def test_run_supervisor_delegates_then_finishes():
    supervisor_model = ScriptedModel(
        "fake-1",
        [
            _text_response("DELEGATE: worker1 :: do the subtask", "s1"),
            _text_response("DONE: final answer here", "s2"),
        ],
    )
    supervisor = Agent("supervisor", supervisor_model)

    worker_model = ScriptedModel("fake-1", [_text_response("subtask done", "w1")])
    worker = Agent("worker1", worker_model)

    result = run_supervisor(supervisor, {"worker1": worker}, "big task")

    assert result["final_answer"] == "final answer here"
    assert result["results"]["worker1"] == "subtask done"


def test_run_supervisor_records_error_for_unknown_worker_and_continues():
    supervisor_model = ScriptedModel(
        "fake-1",
        [
            _text_response("DELEGATE: ghost :: do it", "s1"),
            _text_response("DONE: gave up on ghost", "s2"),
        ],
    )
    supervisor = Agent("supervisor", supervisor_model)

    result = run_supervisor(supervisor, {}, "task")

    assert "unknown worker" in result["results"]["ghost"]
    assert result["final_answer"] == "gave up on ghost"


def test_run_swarm_hands_off_between_agents():
    a_model = ScriptedModel("fake-1", [_text_response("HANDOFF: b :: pass to b", "a1")])
    a = Agent("a", a_model)
    b_model = ScriptedModel("fake-1", [_text_response("final from b", "b1")])
    b = Agent("b", b_model)

    result = run_swarm({"a": a, "b": b}, "a", "task")

    assert result["final_answer"] == "final from b"
    assert result["trace"][0]["agent"] == "a"
    assert result["trace"][1]["agent"] == "b"


def test_agent_node_builds_a_dynamic_cyclic_multi_agent_graph():
    # the exact shape a fixed pattern (sequential/supervisor/parallel/
    # swarm) can't express: draft -> validate -> (loop back to revise if
    # it fails, keeping draft's own output untouched) -> validate again.
    writer = Agent("writer", ScriptedModel("fake-1", [_text_response("draft v1", "w1")]))
    validator = ScriptedModel(
        "fake-1",
        [_text_response("FAIL: too short", "v1"), _text_response("PASS", "v2")],
    )
    validator_agent = Agent("validator", validator)
    reviser = Agent("reviser", ScriptedModel("fake-1", [_text_response("draft v2, much longer", "r1")]))

    graph = Graph(entry="draft")
    graph.add_node("draft", agent_node(writer))
    graph.add_node("validate", agent_node(validator_agent, input_key="draft_output"))
    graph.add_node("revise", agent_node(reviser, input_key="draft_output", output_key="draft_output"))
    graph.add_edge("draft", "validate")
    graph.add_conditional_edges(
        "validate", lambda state: "revise" if "FAIL" in state["validator_output"] else END
    )
    graph.add_edge("revise", "validate")

    result = run_graph(graph, {})

    assert result.state["draft_output"] == "draft v2, much longer"
    assert result.state["validator_output"] == "PASS"
    assert result.history == ["draft", "validate", "revise", "validate"]
