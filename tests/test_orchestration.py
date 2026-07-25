from kel.agents import Agent, run_parallel, run_supervisor, run_swarm, sequential_pipeline
from kel.models.types import ModelResponse, TextPart, Usage
from kel.runtime import run_graph
from helpers import ScriptedModel


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
