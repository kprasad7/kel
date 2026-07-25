"""Fast-tier routers (DESIGN.md 3.14, descoped per §7): rules and
embedding-similarity, deliberately *not* a custom-trained model — that
needs a training pipeline and real traffic volume this project doesn't
have on day one. Both expose `predict_route(...) -> Route | None`
(`None` means "no confident match, ask the slow tier"), the tiny
interface a genuinely learned router could later drop in behind."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kel.brain.types import Route


@dataclass
class Rule:
    predicate: Callable[[dict[str, Any]], bool]
    target: str
    confidence: float = 1.0
    reason: str = "rule matched"


class RuleRouter:
    def __init__(self, rules: list[Rule] | None = None, *, default: str | None = None, default_confidence: float = 0.3):
        self.rules: list[Rule] = list(rules or [])
        self.default = default
        self.default_confidence = default_confidence

    def add_rule(self, predicate: Callable[[dict[str, Any]], bool], target: str, confidence: float = 1.0) -> None:
        self.rules.append(Rule(predicate=predicate, target=target, confidence=confidence))

    def predict_route(self, state: dict[str, Any]) -> Route | None:
        for rule in self.rules:
            if rule.predicate(state):
                return Route(target=rule.target, confidence=rule.confidence, tier="fast", reason=rule.reason)
        if self.default is not None:
            return Route(target=self.default, confidence=self.default_confidence, tier="fast", reason="default fallback")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


class EmbeddingRouter:
    """Nearest-neighbor routing over a small set of labeled examples —
    "if this looks like past query X, route the way X was routed." Grows
    more useful as more examples are added over time (e.g. fed by
    kel.heal's failure->fix history), without ever requiring a training
    step."""

    def __init__(self, embedder: Callable[[str], list[float]], examples: list[tuple[str, str]] | None = None):
        self.embedder = embedder
        self._examples: list[tuple[list[float], str]] = []
        for text, target in examples or []:
            self.add_example(text, target)

    def add_example(self, text: str, target: str) -> None:
        self._examples.append((self.embedder(text), target))

    def predict_route(self, query_text: str) -> Route | None:
        if not self._examples:
            return None
        query_vec = self.embedder(query_text)
        best_score, best_target = max(
            ((_cosine_similarity(query_vec, vec), target) for vec, target in self._examples),
            key=lambda pair: pair[0],
        )
        return Route(target=best_target, confidence=best_score, tier="fast", reason="nearest example match")
