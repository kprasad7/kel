from fastapi import FastAPI
from fastapi.testclient import TestClient

from helpers import ScriptedModel, ScriptedStreamModel
from kel.agents.agent import Agent
from kel.models.types import MessageStop, ModelResponse, TextDelta, TextPart, Usage
from kel.sdk.fastapi_adapter import add_agent_routes, create_fastapi_app


def _agent_with_response(text: str, replies: int = 1) -> Agent:
    response = ModelResponse(id="r", model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())
    model = ScriptedModel("fake-1", [response] * replies)
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


def test_agent_factory_gives_each_session_id_its_own_conversation_history():
    created = []

    def factory() -> Agent:
        agent = _agent_with_response("hi", replies=2)
        created.append(agent)
        return agent

    app = create_fastapi_app(factory)
    client = TestClient(app)

    client.post("/invoke", json={"input": "hello", "session_id": "alice"})
    client.post("/invoke", json={"input": "hello", "session_id": "bob"})
    client.post("/invoke", json={"input": "again", "session_id": "alice"})

    assert len(created) == 2  # alice's second call reused her agent, not a new one
    alice_roles = [m.role for m in created[0].memory.working.messages]
    bob_roles = [m.role for m in created[1].memory.working.messages]
    assert alice_roles == ["user", "assistant", "user", "assistant"]
    assert bob_roles == ["user", "assistant"]


def test_agent_factory_without_session_id_reuses_a_single_default_session():
    created = []

    def factory() -> Agent:
        agent = _agent_with_response("hi", replies=2)
        created.append(agent)
        return agent

    app = create_fastapi_app(factory)
    client = TestClient(app)

    client.post("/invoke", json={"input": "one"})
    client.post("/invoke", json={"input": "two"})

    assert len(created) == 1


def test_agent_factory_evicts_a_session_idle_past_the_ttl():
    import time

    created = []

    def factory() -> Agent:
        agent = _agent_with_response("hi")
        created.append(agent)
        return agent

    app = create_fastapi_app(factory, session_ttl_seconds=0.01)
    client = TestClient(app)

    client.post("/invoke", json={"input": "hello", "session_id": "alice"})
    time.sleep(0.05)
    client.post("/invoke", json={"input": "hello again", "session_id": "alice"})

    assert len(created) == 2  # the idle session was evicted, so "alice" got a fresh Agent
