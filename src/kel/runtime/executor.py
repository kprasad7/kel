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


def fork_from_checkpoint(
    graph: Graph,
    checkpoint: Checkpoint,
    *,
    state_overrides: dict[str, Any] | None = None,
    history: list[str] | None = None,
    run_id: str | None = None,
    loop: Loop | None = None,
    tracer: Tracer | None = None,
    checkpoint_store: CheckpointStore | None = None,
) -> GraphRun:
    """Rewind to an arbitrary historical checkpoint and continue execution
    forward from there — a new timeline branching off the old one (a
    fresh `run_id` by default), not merely resuming the same run.

    This is the actual "time travel" `Checkpoint`'s own docstring
    describes ("a run can pause/resume/branch from any point"), which
    `resume_graph` alone doesn't expose: `resume_graph` can only continue
    from an `Interrupt`'s pause point, not an arbitrary earlier step
    pulled from `CheckpointStore.history(run_id)`.

    `state_overrides` patches the rewound state before continuing — e.g.
    fix the broken variable that made step 12 fail — merged on top of the
    checkpoint's own state:

    ```python
    checkpoints = checkpoint_store.history(run_id)
    step_11 = next(c for c in checkpoints if c.step == 11)
    forked = fork_from_checkpoint(graph, step_11, state_overrides={"retry_count": 0})
    ```
    """
    graph.validate()
    state = dict(checkpoint.state)
    if state_overrides:
        state.update(state_overrides)
    return _run(
        graph,
        state,
        current_layer=graph.next_nodes(checkpoint.node, state),
        history=list(history) if history is not None else [checkpoint.node],
        run_id=run_id or uuid.uuid4().hex,
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
            for _node_name in current_layer:
                loop.step()

            # run this layer's nodes concurrently against a snapshot of
            # state, then merge results back in deterministic (layer) order
            snapshot = dict(state)
            futures = {name: pool.submit(graph.nodes[name], dict(snapshot)) for name in current_layer}

            next_layer: list[str] = []
            for node_name in current_layer:
                fallback_node: str | None = None
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
                except Exception as exc:
                    # a plain node crash used to always crash the whole
                    # run — no fallback path, unlike a DAG engine with
                    # per-node error routing. If a fallback was registered
                    # for this node (Graph.set_fallback), route there
                    # instead, with the error captured into state for the
                    # fallback node to actually see and react to.
                    fallback_node = graph.fallback_for(node_name)
                    if fallback_node is None:
                        raise
                    update = {"__error__": {"node": node_name, "error": str(exc)}}

                signature = update.pop("__signature__", None)
                if signature is not None:
                    loop.record_action(signature)

                state.update(update)
                history.append(node_name)

                if checkpoint_store is not None:
                    checkpoint_store.save(
                        Checkpoint(run_id=run_id, step=len(history), node=node_name, state=dict(state))
                    )

                if fallback_node is not None:
                    next_layer.append(fallback_node)
                else:
                    next_layer.extend(graph.next_nodes(node_name, state))

            # de-dupe while preserving order, so the same node requested by
            # two branches in one layer only runs once in the next layer
            current_layer = list(dict.fromkeys(next_layer))

    return GraphRun(run_id=run_id, state=state, history=history)
