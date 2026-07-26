"""FastAPI adapter for `kel.agents.Agent` — the "real deploy story"
`kel.sdk.serve`'s own docstring flagged as a reasonable future upgrade
(ASGI, streaming, auth) behind the same one-line pattern, the way
LangChain's LangServe wraps a runnable as FastAPI routes. Requires
`pip install kel[fastapi]` — FastAPI/Starlette aren't required by kel
core.

Mounts two routes onto any FastAPI app (your own, or a fresh one
`create_fastapi_app` builds for you):

- `POST {prefix}/invoke` — `{"input": "..."}` -> `{"text": ..., "stop_reason": ...}`,
  the same contract `kel.sdk.serve` uses, but real async ASGI instead of
  a blocking stdlib `http.server` thread.
- `POST {prefix}/stream` — `{"input": "..."}` -> Server-Sent Events, one
  per `Agent.run_stream()` event, for a real streaming UI (a browser
  `EventSource`, or any SSE client) without hand-writing the async
  network loop yourself.

Uses `Agent.arun()`/`arun_stream()` (the async methods), not the sync
`run()`/`run_stream()`, so a slow model call doesn't block the whole ASGI
event loop the way a sync call would from inside an async route handler.

Same "interfaces made concrete, not a production framework" shape as
`kel.sdk.serve`/`serve_websocket`: no auth, no rate limiting built in —
mount these routes on an app that already has whatever middleware your
deployment needs, same as you'd do for any other FastAPI route.

**Single-agent mode vs. multi-session mode.** Pass an `Agent` instance
and every request shares it — fine for a single-conversation demo, but
every caller reads/writes the *same* conversation history. For real
multi-user production traffic, pass a zero-arg factory (`lambda: Agent(...)`)
instead: routes then look up `payload["session_id"]` in an internal
registry, lazily building one `Agent` per session (each with its own
memory) and evicting agents idle past `session_ttl_seconds`. `Agent`
already serializes concurrent calls on itself, so concurrent requests for
the same session are safe either way — the factory just stops different
sessions from sharing history.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

from kel.agents.agent import Agent
from kel.sdk._stream_events import event_to_json

AgentFactory = Callable[[], Agent]


class _SessionAgents:
    """Lazily-built, TTL-evicted per-session `Agent` registry: each
    `session_id` gets its own `Agent` (its own conversation history),
    created from `factory` on first use. Idle sessions are evicted after
    `ttl_seconds` so a long-running server doesn't accumulate an Agent
    (and its memory) forever for every session_id that ever connected."""

    def __init__(self, factory: AgentFactory, *, ttl_seconds: float) -> None:
        self._factory = factory
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._agents: dict[str, Agent] = {}
        self._last_used: dict[str, float] = {}

    def get(self, session_id: str) -> Agent:
        now = time.monotonic()
        with self._lock:
            expired = [sid for sid, last in self._last_used.items() if now - last > self._ttl_seconds]
            for sid in expired:
                self._agents.pop(sid, None)
                self._last_used.pop(sid, None)

            agent = self._agents.get(session_id)
            if agent is None:
                agent = self._factory()
                self._agents[session_id] = agent
            self._last_used[session_id] = now
            return agent


def _import_fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import Body, FastAPI
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "The fastapi package is required for the FastAPI adapter. Install it with `pip install kel[fastapi]`."
        ) from exc
    return FastAPI, Body, StreamingResponse


def add_agent_routes(
    app: Any,
    agent: Agent | AgentFactory,
    *,
    prefix: str = "",
    session_ttl_seconds: float = 3600.0,
) -> None:
    """Adds `{prefix}/invoke` and `{prefix}/stream` routes for `agent`
    onto an existing FastAPI `app` — use this instead of
    `create_fastapi_app` when you already have a FastAPI app (with its
    own middleware, auth, other routes) and just want to mount an agent
    onto it.

    `agent` can be a single `Agent` (shared across all requests) or a
    zero-arg factory, in which case each `payload["session_id"]` gets its
    own lazily-created `Agent`, evicted after `session_ttl_seconds` idle."""
    _, Body, StreamingResponse = _import_fastapi()

    if isinstance(agent, Agent):
        sessions: _SessionAgents | None = None
        single_agent: Agent | None = agent
    else:
        sessions = _SessionAgents(agent, ttl_seconds=session_ttl_seconds)
        single_agent = None

    def _resolve(payload: dict) -> Agent:
        if single_agent is not None:
            return single_agent
        assert sessions is not None
        return sessions.get(str(payload.get("session_id", "default")))

    @app.post(f"{prefix}/invoke")
    async def invoke(payload: dict = Body(...)) -> dict:  # noqa: B008 (FastAPI's own DI convention)
        resolved = _resolve(payload)
        response = await resolved.arun(payload.get("input", ""))
        return {"text": response.text, "stop_reason": response.stop_reason}

    @app.post(f"{prefix}/stream")
    async def stream(payload: dict = Body(...)) -> Any:  # noqa: B008
        resolved = _resolve(payload)

        async def event_source():
            async for event in resolved.arun_stream(payload.get("input", "")):
                yield f"data: {json.dumps(event_to_json(event))}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")


def create_fastapi_app(
    agent: Agent | AgentFactory,
    *,
    prefix: str = "",
    session_ttl_seconds: float = 3600.0,
) -> Any:
    """Convenience: builds a fresh `FastAPI` app with `agent` mounted at
    `prefix`, for the common case of "I just want an app to run with
    uvicorn," not integrating into an app you already have. See
    `add_agent_routes` for the single-agent-vs-per-session-factory
    tradeoff."""
    FastAPI, _, _ = _import_fastapi()
    app = FastAPI()
    add_agent_routes(app, agent, prefix=prefix, session_ttl_seconds=session_ttl_seconds)
    return app
