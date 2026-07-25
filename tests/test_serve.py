import json
import urllib.request

from kel.agents.agent import Agent
from kel.models.types import ModelResponse, TextPart, Usage
from kel.sdk.serve import serve
from helpers import ScriptedModel


def _agent_with_response(text: str) -> Agent:
    model = ScriptedModel(
        "fake-1", [ModelResponse(id="r", model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())]
    )
    return Agent("served-agent", model)


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_serve_invoke_returns_agent_response():
    agent = _agent_with_response("served response")
    server = serve(agent, port=0)
    try:
        status, body = _post(server.port, "/invoke", {"input": "hello"})
        assert status == 200
        assert body["text"] == "served response"
        assert body["stop_reason"] == "end_turn"
    finally:
        server.stop()


def test_serve_unknown_path_returns_404():
    agent = _agent_with_response("x")
    server = serve(agent, port=0)
    try:
        status, body = _post(server.port, "/nope", {"input": "hello"})
        assert status == 404
    finally:
        server.stop()


def test_serve_as_context_manager_stops_cleanly():
    agent = _agent_with_response("ctx response")
    with serve(agent, port=0) as server:
        status, body = _post(server.port, "/invoke", {"input": "hi"})
        assert status == 200
        assert body["text"] == "ctx response"
