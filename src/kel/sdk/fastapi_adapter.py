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
"""

from __future__ import annotations

import json
from typing import Any

from kel.agents.agent import Agent
from kel.sdk._stream_events import event_to_json


def _import_fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import Body, FastAPI
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "The fastapi package is required for the FastAPI adapter. Install it with `pip install kel[fastapi]`."
        ) from exc
    return FastAPI, Body, StreamingResponse


def add_agent_routes(app: Any, agent: Agent, *, prefix: str = "") -> None:
    """Adds `{prefix}/invoke` and `{prefix}/stream` routes for `agent`
    onto an existing FastAPI `app` — use this instead of
    `create_fastapi_app` when you already have a FastAPI app (with its
    own middleware, auth, other routes) and just want to mount an agent
    onto it."""
    _, Body, StreamingResponse = _import_fastapi()

    @app.post(f"{prefix}/invoke")
    async def invoke(payload: dict = Body(...)) -> dict:  # noqa: B008 (FastAPI's own DI convention)
        response = await agent.arun(payload.get("input", ""))
        return {"text": response.text, "stop_reason": response.stop_reason}

    @app.post(f"{prefix}/stream")
    async def stream(payload: dict = Body(...)) -> Any:  # noqa: B008
        async def event_source():
            async for event in agent.arun_stream(payload.get("input", "")):
                yield f"data: {json.dumps(event_to_json(event))}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")


def create_fastapi_app(agent: Agent, *, prefix: str = "") -> Any:
    """Convenience: builds a fresh `FastAPI` app with `agent` mounted at
    `prefix`, for the common case of "I just want an app to run with
    uvicorn," not integrating into an app you already have."""
    FastAPI, _, _ = _import_fastapi()
    app = FastAPI()
    add_agent_routes(app, agent, prefix=prefix)
    return app
