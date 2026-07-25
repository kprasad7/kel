from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.brain.types import Route
from kel.observability.tracer import Tracer, get_tracer

FastTier = Callable[[Any], "Route | None"]
SlowTier = Callable[[Any], Route]


class Brain:
    """Two-tier router (DESIGN.md 3.14): try the fast tier first; escalate
    to the slow tier (typically an LLM call) only when the fast tier has
    no confident answer. Every decision is traced — which tier answered,
    at what confidence — so a bad routing call is debuggable."""

    def __init__(
        self,
        *,
        fast_tier: FastTier | None = None,
        slow_tier: SlowTier | None = None,
        confidence_threshold: float = 0.6,
    ):
        self.fast_tier = fast_tier
        self.slow_tier = slow_tier
        self.confidence_threshold = confidence_threshold

    def route(self, state: Any, *, tracer: Tracer | None = None) -> Route:
        tracer = tracer or get_tracer()
        with tracer.span("kel.brain.route") as span:
            fast_route: Route | None = None
            if self.fast_tier is not None:
                fast_route = self.fast_tier(state)
                if fast_route is not None:
                    span.set_attribute("fast_confidence", fast_route.confidence)
                if fast_route is not None and fast_route.confidence >= self.confidence_threshold:
                    span.set_attribute("decided_tier", "fast")
                    span.set_attribute("target", fast_route.target)
                    return fast_route

            if self.slow_tier is not None:
                slow_route = self.slow_tier(state)
                span.set_attribute("decided_tier", "slow")
                span.set_attribute("target", slow_route.target)
                return slow_route

            if fast_route is not None:
                span.set_attribute("decided_tier", "fast_low_confidence_fallback")
                span.set_attribute("target", fast_route.target)
                return fast_route

            raise RuntimeError(
                "Brain could not produce a route: fast tier found no match and no slow tier is configured"
            )
