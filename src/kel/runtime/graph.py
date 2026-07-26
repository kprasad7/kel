"""Typed execution DAG (DESIGN.md 3.10): nodes are plain functions over a
shared state dict, edges can be static, conditional, or cyclic (a router
can return the name of a node already visited — that's how agentic loops
are expressed, not a special case).

Node functions return a partial state update (a dict merged into state),
not the full state — same convention as React/Redux-style reducers, so a
node only has to describe what it changed. A node can optionally include
`"__signature__"` in its return value to opt into stuck-loop detection
(see kel.runtime.executor) — revisiting the same *node* repeatedly is
normal for a cyclic graph, so plain node-name repetition is never treated
as "stuck" on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

NodeFn = Callable[[dict[str, Any]], "dict[str, Any] | None"]
Router = Callable[[dict[str, Any]], "str | list[str]"]

END = "__end__"


class Graph:
    def __init__(self, *, entry: str):
        self.entry = entry
        self.nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, Router] = {}
        self._fallbacks: dict[str, str] = {}

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str) -> None:
        def _router(_state: dict[str, Any], _to: str = to_node) -> list[str]:
            return [_to]

        self._edges[from_node] = _router

    def add_conditional_edges(self, from_node: str, router: Router) -> None:
        self._edges[from_node] = router

    def set_finish(self, name: str) -> None:
        self._edges[name] = lambda _state: []

    def set_fallback(self, node_name: str, fallback_node: str) -> None:
        """If `node_name`'s function raises (anything other than
        `Interrupt`, which pauses the run instead), route to
        `fallback_node` next instead of letting the exception crash the
        whole run — the executor's normal edges/router for `node_name`
        are skipped entirely in favor of going straight to the fallback.
        The exception is captured into `state["__error__"]` as
        `{"node": node_name, "error": str(exc)}` so the fallback node can
        inspect what went wrong (e.g. retry with a different prompt, ask
        a human, or degrade gracefully) rather than only knowing *that*
        something failed."""
        self._fallbacks[node_name] = fallback_node

    def fallback_for(self, node_name: str) -> str | None:
        return self._fallbacks.get(node_name)

    def next_nodes(self, from_node: str, state: dict[str, Any]) -> list[str]:
        router = self._edges.get(from_node)
        if router is None:
            return []
        result = router(state)
        names = [result] if isinstance(result, str) else list(result)
        return [n for n in names if n != END]

    def validate(self) -> None:
        if self.entry not in self.nodes:
            raise ValueError(f"entry node {self.entry!r} is not registered")
        for node_name, fallback_node in self._fallbacks.items():
            if fallback_node not in self.nodes:
                raise ValueError(
                    f"fallback node {fallback_node!r} (registered via set_fallback({node_name!r}, "
                    f"{fallback_node!r})) is not a registered node — add_node it, or fix the typo. "
                    f"Without this check, a failing {node_name!r} would only surface a confusing "
                    f"raw KeyError deep inside the executor once the fallback path was actually taken."
                )
