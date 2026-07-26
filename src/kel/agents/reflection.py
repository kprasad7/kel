"""Reflection / self-correction loop: run an agent, have a critic judge
its output, and if the critic isn't satisfied, feed the critique back to
the agent as context and retry. This is the "reverse feedback" pattern
real complex agentic flows need — a downstream evaluation stage
correcting an upstream generation stage — and it's already fully
expressible today as a cyclic graph (`agent_node()` + `kel.runtime.Graph`
conditional edges routing "revise" back to "validate"). This module is
that same pattern packaged as a ready-made helper for the common case,
so you don't have to hand-wire a `Graph` every time you want one
generator/critic feedback loop.

Not wired into `Agent` automatically — same "composable, not forced"
shape as `HallucinationChecker`/`LLMReranker`. `critic` can be anything:
a `HallucinationChecker`-backed check, an `Agent` acting as a judge, a
plain rule, or (as shown here) any `Callable[[str], tuple[bool, str]]`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from kel.agents.agent import Agent
from kel.models.types import ModelResponse

Critic = Callable[[str], tuple[bool, str]]
AsyncCritic = Callable[[str], Awaitable[tuple[bool, str]]]


@dataclass
class ReflectionResult:
    response: ModelResponse
    attempts: int
    accepted: bool
    """False means `max_attempts` was reached without the critic ever
    accepting the response — `response` is still the last attempt's
    result, not a failure/exception, so the caller decides what "give up"
    means for their use case (return it anyway, escalate, raise)."""


def _feedback_prompt(original_input: str, previous_response: str, feedback: str) -> str:
    return (
        f"Your previous answer was:\n{previous_response}\n\n"
        f"Feedback on that answer: {feedback}\n\n"
        f"Please revise your answer to address this feedback. "
        f"Original request: {original_input}"
    )


def reflect_and_retry(agent: Agent, user_input: str, *, critic: Critic, max_attempts: int = 3) -> ReflectionResult:
    """`critic(response_text) -> (accepted, feedback)` judges each
    attempt. If not accepted, `feedback` is fed back to `agent` as the
    next turn's input (the reverse-feedback step: the critic, downstream
    in the flow, corrects the generator, upstream) and it retries, up to
    `max_attempts` total attempts."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    current_input = user_input
    response: ModelResponse | None = None
    for attempt in range(1, max_attempts + 1):
        response = agent.run(current_input)
        accepted, feedback = critic(response.text)
        if accepted:
            return ReflectionResult(response=response, attempts=attempt, accepted=True)
        current_input = _feedback_prompt(user_input, response.text, feedback)
    assert response is not None
    return ReflectionResult(response=response, attempts=max_attempts, accepted=False)


async def areflect_and_retry(
    agent: Agent, user_input: str, *, critic: AsyncCritic, max_attempts: int = 3
) -> ReflectionResult:
    """Async equivalent of `reflect_and_retry`, using `agent.arun()` and
    an async `critic`."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    current_input = user_input
    response: ModelResponse | None = None
    for attempt in range(1, max_attempts + 1):
        response = await agent.arun(current_input)
        accepted, feedback = await critic(response.text)
        if accepted:
            return ReflectionResult(response=response, attempts=attempt, accepted=True)
        current_input = _feedback_prompt(user_input, response.text, feedback)
    assert response is not None
    return ReflectionResult(response=response, attempts=max_attempts, accepted=False)
