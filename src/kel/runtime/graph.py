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

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._edges[from_node] = lambda _state, _to=to_node: [_to]

    def add_conditional_edges(self, from_node: str, router: Router) -> None:
        self._edges[from_node] = router

    def set_finish(self, name: str) -> None:
        self._edges[name] = lambda _state: []

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
