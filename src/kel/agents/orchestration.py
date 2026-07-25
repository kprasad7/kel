"""Multi-agent orchestration patterns (DESIGN.md 3.11): sequential
pipeline, supervisor/worker, parallel fan-out with merge, swarm (peer
handoff). Each pattern shares state through a plain dict — the "shared
context bus" is not a new abstraction, it's just letting every agent
read/write the same dict instead of only seeing the last message, which is
what directly targets the "agent 3 invents context agent 2 never
produced" failure mode from DESIGN.md 1.2.

Per-agent credentials (own API key/provider vs. one shared key) aren't
special-cased here at all — that's just whatever `ChatModel` each `Agent`
was constructed with; these orchestration functions only ever call
`agent.run(...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kel.agents.agent import Agent
from kel.context.loop import Loop
from kel.runtime.graph import Graph


def sequential_pipeline(agents: list[Agent]) -> Graph:
    """Each agent runs after the previous, seeing every prior agent's
    output via shared graph state (`{agent_name}_output`), not just the
    latest message."""
    if not agents:
        raise ValueError("sequential_pipeline requires at least one agent")

    graph = Graph(entry=agents[0].name)

    def make_node(agent: Agent) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def node(state: dict[str, Any]) -> dict[str, Any]:
            upstream = "\n".join(f"[{k}]: {v}" for k, v in state.items() if k.endswith("_output"))
            task = state.get("input", "")
            combined = f"{upstream}\n\n{task}".strip() if upstream else task
            response = agent.run(combined)
            return {f"{agent.name}_output": response.text}

        return node

    for i, agent in enumerate(agents):
        graph.add_node(agent.name, make_node(agent))
        if i + 1 < len(agents):
            graph.add_edge(agent.name, agents[i + 1].name)
    graph.set_finish(agents[-1].name)
    return graph


def run_parallel(
    agents: dict[str, Agent], task: str, *, merge: Callable[[dict[str, str]], str] | None = None
) -> dict[str, Any]:
    """Fan the same task out to every agent concurrently, then merge."""
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futures = {name: pool.submit(agent.run, task) for name, agent in agents.items()}
        results = {name: future.result().text for name, future in futures.items()}
    merged = merge(results) if merge else "\n\n".join(f"[{k}]: {v}" for k, v in results.items())
    return {"results": results, "merged": merged}


def run_supervisor(supervisor: Agent, workers: dict[str, Agent], task: str, *, max_rounds: int = 5) -> dict[str, Any]:
    """Supervisor decides each round whether to delegate to a named worker
    or finish, via a small text protocol (`DELEGATE: name :: instructions`
    / `DONE: answer`). Bounded by Loop, same as a single agent's tool
    loop — a stuck supervisor (repeating the same delegation) is caught
    the same way a stuck tool-calling agent is."""
    shared_state: dict[str, Any] = {"task": task, "results": {}}
    loop = Loop(max_iterations=max_rounds)

    while True:
        loop.step()
        instruction = (
            f"Task: {shared_state['task']}\n"
            f"Available workers: {', '.join(workers)}\n"
            f"Results so far: {shared_state['results']}\n"
            "Respond with exactly one line: either 'DELEGATE: <worker> :: <instructions>' "
            "or 'DONE: <final answer>'."
        )
        decision = supervisor.run(instruction).text.strip()

        if decision.upper().startswith("DONE:"):
            loop.record_action("DONE")
            shared_state["final_answer"] = decision[len("DONE:") :].strip()
            return shared_state

        if decision.upper().startswith("DELEGATE:"):
            rest = decision[len("DELEGATE:") :].strip()
            worker_name, _, instructions = rest.partition("::")
            worker_name = worker_name.strip()
            loop.record_action(f"DELEGATE:{worker_name}")
            worker = workers.get(worker_name)
            if worker is None:
                shared_state["results"][worker_name] = f"error: unknown worker {worker_name!r}"
                continue
            result = worker.run(instructions.strip() or shared_state["task"])
            shared_state["results"][worker_name] = result.text
        else:
            # supervisor didn't follow the protocol; surface its raw text
            # as the final answer rather than looping forever on garbage.
            shared_state["final_answer"] = decision
            return shared_state


def run_swarm(agents: dict[str, Agent], start_agent: str, task: str, *, max_handoffs: int = 5) -> dict[str, Any]:
    """Peer handoff: whichever agent is active can either answer directly
    or hand off to a named peer via `HANDOFF: name :: note`."""
    shared_state: dict[str, Any] = {"task": task, "trace": []}
    loop = Loop(max_iterations=max_handoffs)
    current_name, current_input = start_agent, task

    while True:
        loop.step()
        agent = agents[current_name]
        response = agent.run(current_input).text.strip()
        shared_state["trace"].append({"agent": current_name, "response": response})

        if response.upper().startswith("HANDOFF:"):
            rest = response[len("HANDOFF:") :].strip()
            next_name, _, note = rest.partition("::")
            next_name = next_name.strip()
            loop.record_action(f"HANDOFF:{current_name}->{next_name}")
            if next_name not in agents:
                shared_state["final_answer"] = response
                return shared_state
            current_name, current_input = next_name, note.strip() or task
            continue

        loop.record_action(f"ANSWER:{current_name}")
        shared_state["final_answer"] = response
        return shared_state
