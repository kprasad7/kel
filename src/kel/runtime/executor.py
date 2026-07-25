"""Executes a Graph: BFS over layers of node names, running each layer's
nodes concurrently (real parallel fan-out for I/O-bound work like model
calls — threads are enough here since these are network-bound, not
CPU-bound), checkpointing after every node, and tracing every step."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from kel.context.loop import Loop
from kel.observability.tracer import Tracer, get_tracer
from kel.runtime.checkpoint import Checkpoint, CheckpointStore
from kel.runtime.graph import Graph
from kel.runtime.interrupt import Interrupt


@dataclass
class GraphRun:
    run_id: str
    state: dict[str, Any]
    history: list[str] = field(default_factory=list)
    interrupted: bool = False
    interrupt_payload: Any = None
    pending_node: str | None = None


def run_graph(
    graph: Graph,
    initial_state: dict[str, Any],
    *,
    loop: Loop | None = None,
    tracer: Tracer | None = None,
    checkpoint_store: CheckpointStore | None = None,
    run_id: str | None = None,
) -> GraphRun:
    graph.validate()
    return _run(
        graph,
        dict(initial_state),
        current_layer=[graph.entry],
        history=[],
        run_id=run_id or uuid.uuid4().hex,
        loop=loop or Loop(max_iterations=100),
        tracer=tracer or get_tracer(),
        checkpoint_store=checkpoint_store,
    )


def resume_graph(
    graph: Graph,
    interrupted_run: GraphRun,
    resume_value: Any,
    *,
    loop: Loop | None = None,
    tracer: Tracer | None = None,
    checkpoint_store: CheckpointStore | None = None,
) -> GraphRun:
    """Continue a graph paused by `Interrupt`, injecting `resume_value`
    into state as `state["__resume_value__"]` for the pending node to read.

    Known limitation: `__resume_value__` stays in state after the resumed
    node consumes it — pop it yourself in that node if a stale value from
    an earlier interrupt could otherwise confuse a later one."""
    if not interrupted_run.interrupted or interrupted_run.pending_node is None:
        raise ValueError("this GraphRun is not interrupted; nothing to resume")
    graph.validate()
    state = dict(interrupted_run.state)
    state["__resume_value__"] = resume_value
    return _run(
        graph,
        state,
        current_layer=[interrupted_run.pending_node],
        history=list(interrupted_run.history),
        run_id=interrupted_run.run_id,
        loop=loop or Loop(max_iterations=100),
        tracer=tracer or get_tracer(),
        checkpoint_store=checkpoint_store,
    )


def _run(
    graph: Graph,
    state: dict[str, Any],
    *,
    current_layer: list[str],
    history: list[str],
    run_id: str,
    loop: Loop,
    tracer: Tracer,
    checkpoint_store: CheckpointStore | None,
) -> GraphRun:
    with ThreadPoolExecutor(max_workers=8) as pool:
        while current_layer:
            for node_name in current_layer:
                loop.step()

            # run this layer's nodes concurrently against a snapshot of
            # state, then merge results back in deterministic (layer) order
            snapshot = dict(state)
            futures = {name: pool.submit(graph.nodes[name], dict(snapshot)) for name in current_layer}

            next_layer: list[str] = []
            for node_name in current_layer:
                try:
                    with tracer.span("kel.runtime.node", node=node_name, run_id=run_id):
                        update = futures[node_name].result() or {}
                except Interrupt as exc:
                    # other futures in this layer may still be running in
                    # the background; the pool waits for them on __exit__
                    # but their results are discarded — known v1 tradeoff.
                    return GraphRun(
                        run_id=run_id,
                        state=state,
                        history=history,
                        interrupted=True,
                        interrupt_payload=exc.payload,
                        pending_node=node_name,
                    )

                signature = update.pop("__signature__", None)
                if signature is not None:
                    loop.record_action(signature)

                state.update(update)
                history.append(node_name)

                if checkpoint_store is not None:
                    checkpoint_store.save(
                        Checkpoint(run_id=run_id, step=len(history), node=node_name, state=dict(state))
                    )

                next_layer.extend(graph.next_nodes(node_name, state))

            # de-dupe while preserving order, so the same node requested by
            # two branches in one layer only runs once in the next layer
            seen: set[str] = set()
            current_layer = [n for n in next_layer if not (n in seen or seen.add(n))]

    return GraphRun(run_id=run_id, state=state, history=history)
