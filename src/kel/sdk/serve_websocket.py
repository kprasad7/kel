"""WebSocket streaming endpoint for `Agent.run_stream()` — the concrete,
ready-made piece behind the specific, common case `kel.realtime`'s
documented "interfaces only" scope leaves as manual plumbing: streaming
an agent's response to a browser/client over a WebSocket connection.
Full bidirectional voice/video realtime orchestration remains a real
vendor SDK's job behind `kel.realtime.providers`' Protocols — this covers
the narrower, much more common "stream text/tool events to a connected
client" case with working infrastructure, not another empty interface.

Requires `pip install kel[websockets]` (the `websockets` package) — a
real, widely-used async WebSocket implementation. kel's job here is
wiring `Agent.run_stream()` to it, not reimplementing the WebSocket
protocol (handshake/framing) by hand.

Same "interfaces only, zero extra dependency by default" shape as
`kel.sdk.serve`'s HTTP server: no auth, no reconnect/backpressure
handling — this is "expose an agent's stream over a socket for
local/demo use," not a production realtime deployment story.

**Single-agent mode vs. per-connection mode.** Pass an `Agent` instance
and every connection shares it — `Agent` serializes concurrent calls on
itself (no corruption from two clients connecting at once), but every
client still reads/writes the *same* conversation history. For real
multi-client serving, pass a zero-arg factory (`lambda: Agent(...)`)
instead: a fresh `Agent` (its own memory) is built for each new
connection and discarded when the connection closes — one connection is
already a natural session boundary for a WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

from kel.agents.agent import Agent
from kel.sdk._stream_events import event_to_json

AgentFactory = Callable[[], Agent]


async def _handle_connection(agent_or_factory: Agent | AgentFactory, websocket: Any, *_extra: Any) -> None:
    # *_extra absorbs the legacy two-arg `(websocket, path)` handler
    # signature older `websockets` versions used, so this works across
    # the library's major versions without pinning one specific shape.
    agent = agent_or_factory() if not isinstance(agent_or_factory, Agent) else agent_or_factory
    async for raw_message in websocket:
        try:
            payload = json.loads(raw_message)
            user_input = payload["input"]
        except (json.JSONDecodeError, KeyError, TypeError):
            await websocket.send(json.dumps({"type": "error", "error": 'expected JSON {"input": "..."}'}))
            continue
        try:
            for event in agent.run_stream(user_input):
                await websocket.send(json.dumps(event_to_json(event)))
        except Exception as exc:
            await websocket.send(json.dumps({"type": "error", "error": str(exc)}))


class KelWebSocketServer:
    def __init__(self, agent: Agent | AgentFactory, *, host: str = "127.0.0.1", port: int = 0):
        self.agent = agent
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._actual_port: int | None = None

    @property
    def port(self) -> int:
        if self._actual_port is None:
            raise RuntimeError("server not started yet")
        return self._actual_port

    def start(self) -> None:
        if self._thread is not None:
            return  # idempotent, same as KelServer.start()
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "The websockets package is required for serve_websocket. "
                "Install it with `pip install kel[websockets]`."
            ) from exc

        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _main() -> None:
                self._server = await websockets.serve(
                    lambda ws, *extra: _handle_connection(self.agent, ws, *extra), self._host, self._port
                )
                self._actual_port = self._server.sockets[0].getsockname()[1]
                ready.set()
                await self._server.wait_closed()

            loop.run_until_complete(_main())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("WebSocket server failed to start within 5 seconds")

    def stop(self) -> None:
        if self._server is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> KelWebSocketServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve_websocket(agent: Agent | AgentFactory, *, host: str = "127.0.0.1", port: int = 0) -> KelWebSocketServer:
    server = KelWebSocketServer(agent, host=host, port=port)
    server.start()
    return server
