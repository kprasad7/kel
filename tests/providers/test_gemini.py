from types import SimpleNamespace

from kel import Message, ToolSpec
from kel.models.providers.gemini import GeminiChatModel
from kel.models.types import MessageStop, TextDelta


class FakeModels:
    def __init__(self, result=None, stream_chunks=None):
        self._result = result
        self._stream_chunks = stream_chunks or []
        self.captured = None

    def generate_content(self, **params):
        self.captured = params
        return self._result

    def generate_content_stream(self, **params):
        self.captured = params
        return iter(self._stream_chunks)

    def embed_content(self, **params):
        self.captured = params
        return self._result


class FakeClient:
    def __init__(self, models: FakeModels):
        self.models = models


def test_generate_maps_text_response():
    result = SimpleNamespace(
        response_id="r1",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hello there", function_call=None)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4),
    )
    client = FakeClient(FakeModels(result=result))
    model = GeminiChatModel("gemini-2.5-flash", client=client)

    resp = model.generate([Message.user("hi")], system="be nice", temperature=0.3)

    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert client.models.captured["config"]["system_instruction"] == "be nice"


def test_generate_maps_tool_call_response():
    result = SimpleNamespace(
        response_id="r2",
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=None, function_call=SimpleNamespace(name="search", args={"q": "kel"}))]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2),
    )
    client = FakeClient(FakeModels(result=result))
    model = GeminiChatModel("gemini-2.5-flash", client=client)
    tools = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]

    resp = model.generate([Message.user("search for kel")], tools=tools)

    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].input == {"q": "kel"}


def test_stream_yields_text_deltas_then_message_stop():
    chunks = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Hi", function_call=None)]), finish_reason=None)
            ],
            usage_metadata=None,
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=" there", function_call=None)]), finish_reason="STOP")
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        ),
    ]
    client = FakeClient(FakeModels(stream_chunks=chunks))
    model = GeminiChatModel("gemini-2.5-flash", client=client)

    collected = list(model.stream([Message.user("hi")]))

    assert [e for e in collected if isinstance(e, TextDelta)][0].text == "Hi"
    assert isinstance(collected[-1], MessageStop)
    assert collected[-1].response.text == "Hi there"


async def test_agenerate_uses_aio_namespace():
    result = SimpleNamespace(
        response_id="r3",
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="async hi", function_call=None)]), finish_reason="STOP")],
        usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=2),
    )

    class FakeAsyncModels:
        async def generate_content(self, **params):
            return result

    class FakeAio:
        models = FakeAsyncModels()

    class FakeAsyncClient(FakeClient):
        aio = FakeAio()

    client = FakeAsyncClient(FakeModels())
    model = GeminiChatModel("gemini-2.5-flash", client=client)

    resp = await model.agenerate([Message.user("hi")])
    assert resp.text == "async hi"
