import time

from kel.brain import Brain, EmbeddingRouter, Route, RuleRouter, race_to_finish, should_continue
from kel.budget import BudgetSnapshot
from kel.retrieval import NaiveHashEmbedder


def test_rule_router_matches_first_true_predicate():
    router = RuleRouter()
    router.add_rule(lambda state: state.get("intent") == "billing", "billing_agent", confidence=0.9)
    router.add_rule(lambda state: state.get("intent") == "support", "support_agent", confidence=0.9)

    route = router.predict_route({"intent": "billing"})
    assert route.target == "billing_agent"
    assert route.tier == "fast"


def test_rule_router_returns_none_when_no_rule_matches_and_no_default():
    router = RuleRouter()
    router.add_rule(lambda state: False, "never")
    assert router.predict_route({}) is None


def test_rule_router_falls_back_to_default():
    router = RuleRouter(default="general_agent", default_confidence=0.2)
    route = router.predict_route({"intent": "unknown"})
    assert route.target == "general_agent"
    assert route.confidence == 0.2


def test_embedding_router_returns_none_with_no_examples():
    router = EmbeddingRouter(embedder=NaiveHashEmbedder(dims=32))
    assert router.predict_route("anything") is None


def test_embedding_router_finds_nearest_example():
    router = EmbeddingRouter(embedder=NaiveHashEmbedder(dims=64))
    router.add_example("what is my account balance", "billing_agent")
    router.add_example("my app keeps crashing", "support_agent")

    route = router.predict_route("how much do I owe on my account")
    assert route.target == "billing_agent"


def test_brain_uses_fast_tier_when_confident():
    calls = []

    def fast(state):
        return Route(target="fast_target", confidence=0.9, tier="fast")

    def slow(state):
        calls.append("slow_called")
        return Route(target="slow_target", confidence=1.0, tier="slow")

    brain = Brain(fast_tier=fast, slow_tier=slow, confidence_threshold=0.6)
    route = brain.route({})

    assert route.target == "fast_target"
    assert calls == []  # slow tier never invoked


def test_brain_escalates_to_slow_tier_when_fast_tier_unconfident():
    def fast(state):
        return Route(target="fast_target", confidence=0.2, tier="fast")

    def slow(state):
        return Route(target="slow_target", confidence=1.0, tier="slow")

    brain = Brain(fast_tier=fast, slow_tier=slow, confidence_threshold=0.6)
    route = brain.route({})

    assert route.target == "slow_target"


def test_brain_falls_back_to_low_confidence_fast_route_when_no_slow_tier():
    def fast(state):
        return Route(target="fast_target", confidence=0.1, tier="fast")

    brain = Brain(fast_tier=fast, confidence_threshold=0.6)
    route = brain.route({})
    assert route.target == "fast_target"


def test_brain_raises_when_nothing_can_produce_a_route():
    brain = Brain()
    try:
        brain.route({})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_race_to_finish_returns_first_sufficient_result():
    def fast_branch():
        return "fast result"

    def slow_branch():
        time.sleep(0.3)
        return "slow result"

    result = race_to_finish({"fast": fast_branch, "slow": slow_branch})
    assert result.winner == "fast"
    assert result.result == "fast result"


def test_race_to_finish_skips_insufficient_results():
    def bad():
        return {"ok": False}

    def good():
        time.sleep(0.05)
        return {"ok": True}

    result = race_to_finish({"bad": bad, "good": good}, is_sufficient=lambda r: r["ok"])
    assert result.winner == "good"


def test_should_continue_stops_when_token_reserve_too_low():
    snapshot = BudgetSnapshot(
        tokens_used=9900,
        cost_usd_used=0,
        tool_calls_used=0,
        wall_seconds_used=1,
        tokens_remaining=100,
        cost_usd_remaining=None,
        tool_calls_remaining=None,
        wall_seconds_remaining=None,
    )
    assert should_continue(snapshot, min_tokens_reserve=200) is False


def test_should_continue_allows_when_plenty_of_budget_left():
    snapshot = BudgetSnapshot(
        tokens_used=100,
        cost_usd_used=0,
        tool_calls_used=0,
        wall_seconds_used=1,
        tokens_remaining=9000,
        cost_usd_remaining=5.0,
        tool_calls_remaining=10,
        wall_seconds_remaining=60,
    )
    assert should_continue(snapshot) is True


def test_should_continue_with_no_limits_set_always_true():
    snapshot = BudgetSnapshot(
        tokens_used=1_000_000,
        cost_usd_used=100,
        tool_calls_used=100,
        wall_seconds_used=1000,
        tokens_remaining=None,
        cost_usd_remaining=None,
        tool_calls_remaining=None,
        wall_seconds_remaining=None,
    )
    assert should_continue(snapshot) is True
