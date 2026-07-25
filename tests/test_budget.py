import pytest

from kel.budget import Budget, BudgetedChatModel, BudgetExceededError, BudgetTracker
from kel.models.base import ChatModel
from kel.models.types import Message, ModelResponse, TextPart, Usage


class _FakeModel(ChatModel):
    provider = "anthropic"
    model_id = "claude-sonnet-5"

    def generate(self, messages, **kwargs):
        return ModelResponse(
            id="r1",
            model=self.model_id,
            content=[TextPart(text="hi")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1000, output_tokens=500),
        )

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def agenerate(self, messages, **kwargs):
        return self.generate(messages, **kwargs)


def test_tracker_allows_usage_within_budget():
    tracker = BudgetTracker(Budget(max_tokens=10_000))
    tracker.record_usage(Usage(input_tokens=100, output_tokens=50))
    assert tracker.tokens_used == 150
    assert tracker.snapshot().tokens_remaining == 9850


def test_tracker_raises_on_token_overrun():
    tracker = BudgetTracker(Budget(max_tokens=100))
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_usage(Usage(input_tokens=80, output_tokens=50))
    assert exc_info.value.dimension == "tokens"


def test_tracker_raises_on_cost_overrun():
    tracker = BudgetTracker(Budget(max_cost_usd=0.01))
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_usage(Usage(input_tokens=10, output_tokens=10), cost_usd=0.02)
    assert exc_info.value.dimension == "cost_usd"


def test_tracker_raises_on_tool_call_overrun():
    tracker = BudgetTracker(Budget(max_tool_calls=1))
    tracker.record_tool_call()
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_tool_call()
    assert exc_info.value.dimension == "tool_calls"


def test_tracker_with_no_limits_never_raises():
    tracker = BudgetTracker()
    tracker.record_usage(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    tracker.record_tool_call()
    assert tracker.tokens_used == 2_000_000


def test_budgeted_chat_model_charges_real_cost_from_pricing_table():
    tracker = BudgetTracker(Budget(max_cost_usd=10.0))
    model = BudgetedChatModel(_FakeModel(), tracker)

    model.generate([Message.user("hi")])

    # claude-sonnet-5: $3/1M in, $15/1M out -> 1000*3/1e6 + 500*15/1e6 = 0.003 + 0.0075
    assert tracker.cost_usd_used == pytest.approx(0.0105)
    assert tracker.tokens_used == 1500


def test_budgeted_chat_model_raises_when_call_pushes_over_budget():
    tracker = BudgetTracker(Budget(max_tokens=1000))
    model = BudgetedChatModel(_FakeModel(), tracker)

    with pytest.raises(BudgetExceededError):
        model.generate([Message.user("hi")])


async def test_budgeted_chat_model_agenerate_charges_cost():
    tracker = BudgetTracker(Budget(max_cost_usd=10.0))
    model = BudgetedChatModel(_FakeModel(), tracker)

    await model.agenerate([Message.user("hi")])

    assert tracker.cost_usd_used == pytest.approx(0.0105)
