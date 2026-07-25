from types import SimpleNamespace

from kel import Message, ToolSpec
from kel.models.types import MessageStop, TextDelta, ToolCallDelta
from kel.models.providers.cohere import CohereChatModel, CohereEmbeddingModel


class FakeClient:
    def __init__(self, chat_result=None, stream_events=None):
        self._chat_result = chat_result
        self._stream_events = stream_events or []
        self.captured_params = None

    def chat(self, **params):
        self.captured_params = params
        return self._chat_result

    def chat_stream(self, **params):
        self.captured_params = params
        return iter(self._stream_events)


def test_generate_maps_text_response():
    result = SimpleNamespace(
        id="chat_1",
        finish_reason="COMPLETE",
        message=SimpleNamespace(content=[SimpleNamespace(type="text", text="hello there")], tool_calls=None),
        usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=10, output_tokens=4)),
    )
    client = FakeClient(chat_result=result)
    model = CohereChatModel("command-r-plus", client=client)

    resp = model.generate([Message.user("hi")], system="be nice", temperature=0.3)

    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 4
    params = client.captured_params
    assert params["messages"][0] == {"role": "system", "content": "be nice"}
    assert params["temperature"] == 0.3


def test_generate_maps_tool_call_response():
    result = SimpleNamespace(
        id="chat_2",
        finish_reason="TOOL_CALL",
        message=SimpleNamespace(
            content=[],
            tool_calls=[
                SimpleNamespace(id="call_1", function=SimpleNamespace(name="search", arguments='{"q": "kel"}'))
            ],
        ),
        usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=5, output_tokens=2)),
    )
    client = FakeClient(chat_result=result)
    model = CohereChatModel("command-r-plus", client=client)
    tools = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]

    resp = model.generate([Message.user("search for kel")], tools=tools)

    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].input == {"q": "kel"}
    assert client.captured_params["tools"][0]["function"]["name"] == "search"


def test_stream_accumulates_text_and_tool_call_events():
    events = [
        SimpleNamespace(
            type="content-delta",
            delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="Hi"))),
        ),
        SimpleNamespace(
            type="tool-call-start",
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        index=0, id="call_1", function=SimpleNamespace(name="search", arguments='{"q":')
                    )
                )
            ),
        ),
        SimpleNamespace(
            type="tool-call-delta",
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        index=0, id=None, function=SimpleNamespace(name=None, arguments='"kel"}')
                    )
                )
            ),
        ),
        SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                finish_reason="TOOL_CALL",
                usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=3, output_tokens=2)),
            ),
        ),
    ]
    client = FakeClient(stream_events=events)
    model = CohereChatModel("command-r-plus", client=client)

    collected = list(model.stream([Message.user("search for kel")]))

    text_deltas = [e for e in collected if isinstance(e, TextDelta)]
    assert text_deltas[0].text == "Hi"

    tool_events = [e for e in collected if isinstance(e, ToolCallDelta)]
    assert tool_events[0].tool_call.name == "search"
    assert tool_events[0].tool_call.input == {"q": "kel"}

    stop = collected[-1]
    assert isinstance(stop, MessageStop)
    assert stop.response.stop_reason == "tool_use"
    assert stop.response.usage.input_tokens == 3


class FakeAsyncClient:
    def __init__(self, chat_result=None, stream_events=None):
        self._chat_result = chat_result
        self._stream_events = stream_events or []
        self.captured_params = None

    async def chat(self, **params):
        self.captured_params = params
        return self._chat_result

    def chat_stream(self, **params):
        self.captured_params = params
        return self._aiter_events()

    async def _aiter_events(self):
        for event in self._stream_events:
            yield event


async def test_agenerate_maps_text_response():
    result = SimpleNamespace(
        id="chat_3",
        finish_reason="COMPLETE",
        message=SimpleNamespace(content=[SimpleNamespace(type="text", text="async hello")], tool_calls=None),
        usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=8, output_tokens=3)),
    )
    client = FakeAsyncClient(chat_result=result)
    model = CohereChatModel("command-a-03-2025", async_client=client)

    resp = await model.agenerate([Message.user("hi")])

    assert resp.text == "async hello"
    assert resp.usage.input_tokens == 8


async def test_astream_accumulates_text_and_tool_call_events():
    events = [
        SimpleNamespace(
            type="content-delta",
            delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="Hi"))),
        ),
        SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                finish_reason="COMPLETE",
                usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=1, output_tokens=1)),
            ),
        ),
    ]
    client = FakeAsyncClient(stream_events=events)
    model = CohereChatModel("command-a-03-2025", async_client=client)

    collected = [event async for event in model.astream([Message.user("hi")])]

    text_deltas = [e for e in collected if isinstance(e, TextDelta)]
    assert text_deltas[0].text == "Hi"
    stop = collected[-1]
    assert isinstance(stop, MessageStop)
    assert stop.response.stop_reason == "end_turn"


class FakeEmbedClient:
    def __init__(self, result):
        self._result = result
        self.captured_params = None

    def embed(self, **params):
        self.captured_params = params
        return self._result


def test_cohere_embedding_model_returns_vectors_and_sends_input_type():
    result = SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1, 0.2], [0.3, 0.4]]))
    client = FakeEmbedClient(result)
    model = CohereEmbeddingModel("embed-v4.0", client=client, input_type="search_query")

    vectors = model.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.captured_params["input_type"] == "search_query"
    assert client.captured_params["texts"] == ["hello", "world"]
