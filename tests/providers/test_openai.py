from types import SimpleNamespace

from kel import Message, ToolResultPart, ToolSpec, ToolUsePart
from kel.models.providers.openai import OpenAIChatModel, OpenAIEmbeddingModel
from kel.models.types import MessageStop, Role, TextDelta, ToolCallDelta


class FakeCompletions:
    def __init__(self, result):
        self._result = result
        self.captured_params = None

    def create(self, **params):
        self.captured_params = params
        return self._result


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = FakeChat(completions)


def test_generate_maps_text_response():
    result = SimpleNamespace(
        id="chatcmpl_1",
        model="gpt-5.2",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello there", tool_calls=None), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
    )
    client = FakeClient(FakeCompletions(result))
    model = OpenAIChatModel("gpt-5.2", client=client)

    resp = model.generate([Message.user("hi")], system="be nice", temperature=0.3)

    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    params = client.chat.completions.captured_params
    assert params["messages"][0] == {"role": "system", "content": "be nice"}
    assert params["temperature"] == 0.3


def test_generate_maps_tool_call_response():
    result = SimpleNamespace(
        id="chatcmpl_2",
        model="gpt-5.2",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="search", arguments='{"q": "kel"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )
    client = FakeClient(FakeCompletions(result))
    model = OpenAIChatModel("gpt-5.2", client=client)
    tools = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]

    resp = model.generate([Message.user("search for kel")], tools=tools)

    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].input == {"q": "kel"}
    params = client.chat.completions.captured_params
    assert params["tools"][0]["function"]["name"] == "search"


def test_tool_result_message_becomes_separate_tool_role_message():
    result = SimpleNamespace(
        id="chatcmpl_3",
        model="gpt-5.2",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    client = FakeClient(FakeCompletions(result))
    model = OpenAIChatModel("gpt-5.2", client=client)

    messages = [
        Message(role=Role.ASSISTANT, content=[ToolUsePart(id="call_1", name="search", input={"q": "kel"})]),
        Message(
            role=Role.USER,
            content=[ToolResultPart(tool_use_id="call_1", content="found: kel repo")],
        ),
    ]
    model.generate(messages)

    sent = client.chat.completions.captured_params["messages"]
    assert sent[0]["role"] == "assistant"
    assert sent[0]["tool_calls"][0]["function"]["name"] == "search"
    assert sent[1] == {"role": "tool", "tool_call_id": "call_1", "content": "found: kel repo"}


def test_stream_accumulates_text_and_tool_call_fragments():
    chunks = [
        SimpleNamespace(
            id="chatcmpl_4",
            model="gpt-5.2",
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None), finish_reason=None)],
        ),
        SimpleNamespace(
            id="chatcmpl_4",
            model="gpt-5.2",
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="search", arguments='{"q":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            id="chatcmpl_4",
            model="gpt-5.2",
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='"kel"}'))
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
        SimpleNamespace(
            id="chatcmpl_4",
            model="gpt-5.2",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            choices=[],
        ),
    ]
    client = FakeClient(FakeCompletions(chunks))
    model = OpenAIChatModel("gpt-5.2", client=client)

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


class FakeAsyncCompletions:
    def __init__(self, result):
        self._result = result
        self.captured_params = None

    async def create(self, **params):
        self.captured_params = params
        return self._result


class FakeAsyncChat:
    def __init__(self, completions: FakeAsyncCompletions):
        self.completions = completions


class FakeAsyncClient:
    def __init__(self, completions: FakeAsyncCompletions):
        self.chat = FakeAsyncChat(completions)


async def test_agenerate_maps_text_response():
    result = SimpleNamespace(
        id="chatcmpl_5",
        model="gpt-5.2",
        choices=[SimpleNamespace(message=SimpleNamespace(content="async hi", tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=6, completion_tokens=2),
    )
    client = FakeAsyncClient(FakeAsyncCompletions(result))
    model = OpenAIChatModel("gpt-5.2", async_client=client)

    resp = await model.agenerate([Message.user("hi")])

    assert resp.text == "async hi"
    assert resp.usage.input_tokens == 6


class _FakeAsyncChunkStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


async def test_astream_accumulates_text_and_tool_call_fragments():
    chunks = [
        SimpleNamespace(
            id="chatcmpl_6",
            model="gpt-5.2",
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None), finish_reason=None)],
        ),
        SimpleNamespace(
            id="chatcmpl_6",
            model="gpt-5.2",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")],
        ),
    ]

    class _StreamingFakeCompletions(FakeAsyncCompletions):
        async def create(self, **params):
            self.captured_params = params
            return _FakeAsyncChunkStream(chunks)

    client = FakeAsyncClient(_StreamingFakeCompletions(None))
    model = OpenAIChatModel("gpt-5.2", async_client=client)

    collected = [event async for event in model.astream([Message.user("hi")])]

    text_deltas = [e for e in collected if isinstance(e, TextDelta)]
    assert text_deltas[0].text == "Hi"
    stop = collected[-1]
    assert isinstance(stop, MessageStop)
    assert stop.response.usage.input_tokens == 2


class FakeEmbeddingsClient:
    def __init__(self, result):
        self._result = result
        self.captured_params = None

    class _Embeddings:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **params):
            self._outer.captured_params = params
            return self._outer._result

    @property
    def embeddings(self):
        return self._Embeddings(self)


def test_openai_embedding_model_returns_vectors_in_order():
    result = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ]
    )
    client = FakeEmbeddingsClient(result)
    model = OpenAIEmbeddingModel("text-embedding-3-small", client=client)

    vectors = model.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.captured_params["input"] == ["hello", "world"]
    assert client.captured_params["model"] == "text-embedding-3-small"
