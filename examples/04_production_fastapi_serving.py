"""Serving an Agent over FastAPI with one conversation per session, not
one shared conversation for every caller. Passing a single Agent instance
to create_fastapi_app() means every request reads/writes the same
history — fine for a demo, wrong for real multi-user traffic. Passing a
zero-arg factory instead gives each session_id its own Agent, built
lazily and evicted after being idle (default: 1 hour).

Run:
    pip install "pykel[fastapi]"
    export ANTHROPIC_API_KEY=sk-...
    uvicorn examples.04_production_fastapi_serving:app --reload

Then, from another terminal:
    curl -X POST localhost:8000/invoke -H 'content-type: application/json' \
        -d '{"input": "my name is Alex", "session_id": "alex"}'
    curl -X POST localhost:8000/invoke -H 'content-type: application/json' \
        -d '{"input": "what is my name?", "session_id": "alex"}'
    # -> answers "Alex", because "alex" is a session, not a one-off call

    curl -X POST localhost:8000/invoke -H 'content-type: application/json' \
        -d '{"input": "what is my name?", "session_id": "sam"}'
    # -> "sam" has never been told a name: a separate conversation, as expected
"""

from kel import get_model
from kel.agents import Agent
from kel.sdk import create_fastapi_app


def make_agent() -> Agent:
    # called once per new session_id, not once per request — the Agent
    # (and its Memory) persists across requests within the same session
    return Agent(
        "assistant",
        get_model("anthropic:claude-sonnet-5"),
        system_prompt="Be concise. Remember what the user tells you earlier in the conversation.",
    )


# a factory, not an Agent instance, is what turns on per-session isolation
app = create_fastapi_app(make_agent, session_ttl_seconds=3600)
