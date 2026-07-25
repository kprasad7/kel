from types import SimpleNamespace

from kel import Message, ToolSpec
from kel.models.providers.mistral import MistralChatModel
from kel.models.types import MessageStop, TextDelta, ToolCallDelta


class FakeChat:
    def __init__(self, result=None, stream_events=None):
        self._result = result
        self._stream_events = stream_events or []
        self.captured = None

    def complete(self, **params):
        self.captured = params
        return self._result

    def stream(self, **params):
        self.captured = params
        return iter(self._stream_events)

    async def complete_async(self, **params):
        self.captured = params
        return self._result

    async def stream_async(self, **params):
        self.captured = params
        for event in self._stream_events:
            yield event


class FakeClient:
    def __init__(self, chat: FakeChat):
        self.chat = chat


def test_generate_maps_text_response():
    result = SimpleNamespace(
        id="c1",
        model="mistral-large-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello there", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
    )
    client = FakeClient(FakeChat(result=result))
    model = MistralChatModel("mistral-large-latest", client=client)

    resp = model.generate([Message.user("hi")], system="be nice", temperature=0.3)

    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert client.chat.captured["messages"][0] == {"role": "system", "content": "be nice"}


def test_generate_maps_tool_call_response():
    result = SimpleNamespace(
        id="c2",
        model="mistral-large-latest",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="search", arguments='{"q": "kel"}'))],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    client = FakeClient(FakeChat(result=result))
    model = MistralChatModel("mistral-large-latest", client=client)
    tools = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]

    resp = model.generate([Message.user("search")], tools=tools)

    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].input == {"q": "kel"}


def test_stream_accumulates_text_and_tool_call_fragments():
    events = [
        SimpleNamespace(data=SimpleNamespace(id="c3", choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None), finish_reason=None)], usage=None)),
        SimpleNamespace(data=SimpleNamespace(id="c3", choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))),
    ]
    client = FakeClient(FakeChat(stream_events=events))
    model = MistralChatModel("mistral-large-latest", client=client)

    collected = list(model.stream([Message.user("hi")]))

    assert [e for e in collected if isinstance(e, TextDelta)][0].text == "Hi"
    assert isinstance(collected[-1], MessageStop)


async def test_agenerate_uses_complete_async():
    result = SimpleNamespace(
        id="c4", model="mistral-large-latest",
        choices=[SimpleNamespace(message=SimpleNamespace(content="async hi", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    client = FakeClient(FakeChat(result=result))
    model = MistralChatModel("mistral-large-latest", client=client)

    resp = await model.agenerate([Message.user("hi")])
    assert resp.text == "async hi"


def test_api_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-key-123")
    model = MistralChatModel("mistral-large-latest")
    assert model._api_key == "env-key-123"


def test_explicit_api_key_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-key")
    model = MistralChatModel("mistral-large-latest", api_key="explicit-key")
    assert model._api_key == "explicit-key"
