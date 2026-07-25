import pytest

from kel.brain import EmbeddingRouter
from kel.heal import Diagnosis, HealExhaustedError, Healer, feed_heal_log_into_router, parse_diagnosis
from kel.retrieval import NaiveHashEmbedder


def test_parse_diagnosis_extracts_strategy_and_reason():
    text = "STRATEGY: retry\nREASON: transient network error"
    d = parse_diagnosis(text)
    assert d.strategy == "retry"
    assert d.reason == "transient network error"


def test_parse_diagnosis_defaults_to_escalate_human_when_unparseable():
    d = parse_diagnosis("I have no idea what happened")
    assert d.strategy == "escalate_human"


def test_healer_succeeds_without_any_failure():
    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="retry", reason="n/a"))
    result = healer.run(lambda diagnosis: "ok", idempotent=True)
    assert result == "ok"
    assert healer.log == []


def test_healer_retries_idempotent_call_and_eventually_succeeds():
    calls = {"n": 0}

    def flaky(diagnosis):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "recovered"

    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="retry", reason="transient"), max_attempts=5)
    result = healer.run(flaky, idempotent=True)

    assert result == "recovered"
    assert calls["n"] == 3
    assert len(healer.log) == 2
    assert all(a.outcome == "retried" for a in healer.log)


def test_healer_refuses_to_retry_non_idempotent_action_even_if_diagnosis_says_retry():
    def always_fails(diagnosis):
        raise RuntimeError("payment failed")

    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="retry", reason="looks transient"))

    with pytest.raises(HealExhaustedError) as exc_info:
        healer.run(always_fails, idempotent=False)

    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.attempts[0].outcome == "escalated"
    assert exc_info.value.attempts[0].diagnosis.strategy == "escalate_human"
    assert "non-idempotent" in exc_info.value.attempts[0].diagnosis.reason


def test_healer_escalates_immediately_when_diagnosis_says_escalate_human():
    def always_fails(diagnosis):
        raise RuntimeError("unrecoverable")

    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="escalate_human", reason="unsafe to retry"))

    with pytest.raises(HealExhaustedError):
        healer.run(always_fails, idempotent=True)

    assert len(healer.log) == 1


def test_healer_exhausts_after_max_attempts_of_persistent_failure():
    def always_fails(diagnosis):
        raise RuntimeError("still broken")

    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="retry", reason="try again"), max_attempts=3)

    with pytest.raises(HealExhaustedError) as exc_info:
        healer.run(always_fails, idempotent=True)

    assert len(exc_info.value.attempts) == 3


def test_healer_passes_diagnosis_forward_so_fn_can_adapt():
    seen_strategies = []

    def adaptive(diagnosis):
        seen_strategies.append(diagnosis.strategy if diagnosis else None)
        if diagnosis is None:
            raise RuntimeError("first failure")
        return "adapted"

    healer = Healer(diagnose=lambda exc, ctx: Diagnosis(strategy="fallback_model", reason="switch model"))
    result = healer.run(adaptive, idempotent=True)

    assert result == "adapted"
    assert seen_strategies == [None, "fallback_model"]


def test_feed_heal_log_into_router_lets_router_route_similar_future_errors():
    def always_fails(diagnosis):
        raise RuntimeError("rate limit exceeded on provider X")

    healer = Healer(
        diagnose=lambda exc, ctx: Diagnosis(strategy="escalate_human", reason="rate limited"), max_attempts=1
    )
    with pytest.raises(HealExhaustedError):
        healer.run(always_fails, idempotent=True)

    router = EmbeddingRouter(embedder=NaiveHashEmbedder(dims=64))
    feed_heal_log_into_router(healer.log, router)

    route = router.predict_route("rate limit exceeded on provider Y")
    assert route.target == "escalate_human"
