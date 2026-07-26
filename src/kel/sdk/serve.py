"""LangServe-equivalent one-line deploy (DESIGN.md 3.13): `POST /invoke`
against a running Agent. Built on the stdlib `http.server`, not FastAPI —
zero extra dependency for a genuinely working synchronous HTTP endpoint.
No streaming, no auth, no concurrency tuning: this is "expose an agent for
local testing/demos," not a production deployment story. A real deploy
story (ASGI, streaming, auth) is a reasonable future upgrade behind the
same `serve()` call, not something to fake here.

**Every request shares the one `Agent` (and its memory/conversation)
passed in here.** `Agent` serializes concurrent calls on itself (no
corruption from interleaved requests), but they still all read/write the
*same* conversation history — this is a single-conversation demo
endpoint, not a multi-user API. For real multi-user serving, construct a
fresh `Agent` (with its own `Memory(session_id=..., ...)`) per session
rather than sharing one across every caller."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from kel.agents.agent import Agent


def _make_handler(agent: Agent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (stdlib method name)
            # always drain the request body first, even on a 404 — leaving
            # unread bytes on the socket before responding/closing can
            # abort the client's connection mid-write (observed on Windows).
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"

            if self.path != "/invoke":
                self._respond(404, {"error": "not found, POST to /invoke"})
                return
            try:
                body = json.loads(raw_body or b"{}")
                response = agent.run(body.get("input", ""))
                self._respond(200, {"text": response.text, "stop_reason": response.stop_reason})
            except Exception as exc:
                self._respond(500, {"error": str(exc)})

        def _respond(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # silence default stderr access logging

    return Handler


class KelServer:
    def __init__(self, agent: Agent, *, host: str = "127.0.0.1", port: int = 0):
        self.agent = agent
        self._httpd = HTTPServer((host, port), _make_handler(agent))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        if self._thread is not None:
            return  # idempotent: serve() already starts it, so `with serve(...) as s:` shouldn't double-start
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> KelServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve(agent: Agent, *, host: str = "127.0.0.1", port: int = 0) -> KelServer:
    server = KelServer(agent, host=host, port=port)
    server.start()
    return server
