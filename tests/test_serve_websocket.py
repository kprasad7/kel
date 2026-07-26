import json

import pytest
import websockets

from helpers import ScriptedStreamModel
from kel.agents.agent import Agent
from kel.models.types import MessageStop, ModelResponse, TextDelta, TextPart, Usage
from kel.sdk.serve_websocket import serve_websocket


def _final_response(text: str, rid: str = "r") -> ModelResponse:
    return ModelResponse(id=rid, model="fake-1", content=[TextPart(text=text)], stop_reason="end_turn", usage=Usage())


async def test_serve_websocket_streams_text_deltas_and_message_stop():
    events = [[TextDelta(text="hel"), TextDelta(text="lo"), MessageStop(response=_final_response("hello"))]]
    model = ScriptedStreamModel("fake-1", events)
    agent = Agent("ws-agent", model)

    with serve_websocket(agent, port=0) as server:
        async with websockets.connect(f"ws://127.0.0.1:{server.port}") as client:
            await client.send(json.dumps({"input": "hi"}))

            received = [json.loads(await client.recv()) for _ in range(3)]

    assert received[0] == {"type": "text_delta", "text": "hel"}
    assert received[1] == {"type": "text_delta", "text": "lo"}
    assert received[2]["type"] == "message_stop"
    assert received[2]["text"] == "hello"
    assert received[2]["stop_reason"] == "end_turn"


async def test_serve_websocket_returns_error_for_malformed_payload():
    model = ScriptedStreamModel("fake-1", [])
    agent = Agent("ws-agent", model)

    with serve_websocket(agent, port=0) as server:
        async with websockets.connect(f"ws://127.0.0.1:{server.port}") as client:
            await client.send(json.dumps({"not_input": "oops"}))
            reply = json.loads(await client.recv())

    assert reply["type"] == "error"


async def test_serve_websocket_handles_multiple_messages_on_one_connection():
    events = [
        [MessageStop(response=_final_response("first"))],
        [MessageStop(response=_final_response("second"))],
    ]
    model = ScriptedStreamModel("fake-1", events)
    agent = Agent("ws-agent", model)

    with serve_websocket(agent, port=0) as server:
        async with websockets.connect(f"ws://127.0.0.1:{server.port}") as client:
            await client.send(json.dumps({"input": "one"}))
            first = json.loads(await client.recv())

            await client.send(json.dumps({"input": "two"}))
            second = json.loads(await client.recv())

    assert first["text"] == "first"
    assert second["text"] == "second"


def test_serve_websocket_port_property_raises_before_start():
    from kel.sdk.serve_websocket import KelWebSocketServer

    model = ScriptedStreamModel("fake-1", [])
    agent = Agent("ws-agent", model)
    server = KelWebSocketServer(agent)

    with pytest.raises(RuntimeError):
        _ = server.port
