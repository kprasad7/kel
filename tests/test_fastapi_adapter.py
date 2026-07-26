from fastapi import FastAPI
from fastapi.testclient import TestClient

from helpers import ScriptedModel, ScriptedStreamModel
from kel.agents.agent import Agent
from kel.models.types import MessageStop, ModelResponse, TextDelta, TextPart, Usage
from kel.sdk.fastapi_adapter import add_agent_routes, create_fastapi_app


def _agent_with_response(text: str) -> Agent:
    model = ScriptedModel(
        "fake-1", [ModelResponse(id="r", model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())]
    )
    return Agent("fastapi-agent", model)


def test_create_fastapi_app_invoke_returns_agent_response():
    agent = _agent_with_response("served response")
    app = create_fastapi_app(agent)
    client = TestClient(app)

    resp = client.post("/invoke", json={"input": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"text": "served response", "stop_reason": "end_turn"}


def test_add_agent_routes_mounts_onto_an_existing_app_with_a_prefix():
    agent = _agent_with_response("mounted response")
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    add_agent_routes(app, agent, prefix="/agent")
    client = TestClient(app)

    assert client.get("/health").json() == {"ok": True}
    resp = client.post("/agent/invoke", json={"input": "hi"})
    assert resp.json()["text"] == "mounted response"


def test_stream_route_sends_server_sent_events_for_each_run_stream_event():
    events = [
        [TextDelta(text="hel"), TextDelta(text="lo"), MessageStop(
            response=ModelResponse(id="r", model="fake-1", content=[TextPart(text="hello")], stop_reason="end_turn", usage=Usage())
        )]
    ]
    model = ScriptedStreamModel("fake-1", events)
    agent = Agent("fastapi-agent", model)
    app = create_fastapi_app(agent)
    client = TestClient(app)

    with client.stream("POST", "/stream", json={"input": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    lines = [line for line in body.split("\n\n") if line.strip()]
    assert len(lines) == 3
    assert lines[0] == 'data: {"type": "text_delta", "text": "hel"}'
    assert lines[1] == 'data: {"type": "text_delta", "text": "lo"}'
    assert '"type": "message_stop"' in lines[2]
    assert '"text": "hello"' in lines[2]


def test_invoke_defaults_to_empty_input_when_missing():
    agent = _agent_with_response("default response")
    app = create_fastapi_app(agent)
    client = TestClient(app)

    resp = client.post("/invoke", json={})

    assert resp.status_code == 200
    assert resp.json()["text"] == "default response"
