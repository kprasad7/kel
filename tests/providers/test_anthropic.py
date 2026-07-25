from types import SimpleNamespace

from kel import Message, TextDelta, ToolCallDelta, ToolSpec
from kel.models.types import MessageStop
from kel.models.providers.anthropic import AnthropicChatModel


class FakeMessages:
    def __init__(self, create_result=None, stream_ctx=None):
        self._create_result = create_result
        self._stream_ctx = stream_ctx
        self.captured_params = None

    def create(self, **params):
        self.captured_params = params
        return self._create_result

    def stream(self, **params):
        self.captured_params = params
        return self._stream_ctx


class FakeClient:
    def __init__(self, messages: FakeMessages):
        self.messages = messages


class FakeStreamCtx:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


def test_generate_maps_text_response():
    result = SimpleNamespace(
        id="msg_1",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="text", text="hello there")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )
    client = FakeClient(FakeMessages(create_result=result))
    model = AnthropicChatModel("claude-sonnet-5", client=client)

    resp = model.generate([Message.user("hi")], system="be nice", max_tokens=100, temperature=0.2)

    assert resp.text == "hello there"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 4
    assert client.messages.captured_params["system"] == "be nice"
    assert client.messages.captured_params["temperature"] == 0.2
    assert client.messages.captured_params["messages"][0]["role"] == "user"


def test_generate_maps_tool_use_response():
    result = SimpleNamespace(
        id="msg_2",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="tool_use", id="call_1", name="search", input={"q": "kel"})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
    )
    client = FakeClient(FakeMessages(create_result=result))
    model = AnthropicChatModel("claude-sonnet-5", client=client)
    tools = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]

    resp = model.generate([Message.user("search for kel")], tools=tools)

    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].input == {"q": "kel"}
    assert client.messages.captured_params["tools"][0]["name"] == "search"


def test_stream_yields_text_deltas_then_message_stop():
    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hi")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=" there")),
    ]
    final = SimpleNamespace(
        id="msg_3",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="text", text="Hi there")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = FakeClient(FakeMessages(stream_ctx=FakeStreamCtx(events, final)))
    model = AnthropicChatModel("claude-sonnet-5", client=client)

    collected = list(model.stream([Message.user("hi")]))

    assert [e for e in collected if isinstance(e, TextDelta)][0].text == "Hi"
    assert isinstance(collected[-1], MessageStop)
    assert collected[-1].response.text == "Hi there"


def test_stream_yields_tool_call_delta_for_tool_use_blocks():
    final = SimpleNamespace(
        id="msg_4",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="tool_use", id="call_1", name="search", input={"q": "x"})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = FakeClient(FakeMessages(stream_ctx=FakeStreamCtx([], final)))
    model = AnthropicChatModel("claude-sonnet-5", client=client)

    collected = list(model.stream([Message.user("search")]))

    tool_events = [e for e in collected if isinstance(e, ToolCallDelta)]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.name == "search"


class FakeAsyncMessages:
    def __init__(self, create_result=None, stream_ctx=None):
        self._create_result = create_result
        self._stream_ctx = stream_ctx
        self.captured_params = None

    async def create(self, **params):
        self.captured_params = params
        return self._create_result

    def stream(self, **params):
        self.captured_params = params
        return self._stream_ctx


class FakeAsyncClient:
    def __init__(self, messages: FakeAsyncMessages):
        self.messages = messages


class FakeAsyncStreamCtx:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final


async def test_agenerate_maps_text_response():
    result = SimpleNamespace(
        id="msg_5",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="text", text="async hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    client = FakeAsyncClient(FakeAsyncMessages(create_result=result))
    model = AnthropicChatModel("claude-sonnet-5", async_client=client)

    resp = await model.agenerate([Message.user("hi")])

    assert resp.text == "async hello"
    assert resp.usage.input_tokens == 7


async def test_astream_yields_text_deltas_then_message_stop():
    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hi")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=" async")),
    ]
    final = SimpleNamespace(
        id="msg_6",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="text", text="Hi async")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = FakeAsyncClient(FakeAsyncMessages(stream_ctx=FakeAsyncStreamCtx(events, final)))
    model = AnthropicChatModel("claude-sonnet-5", async_client=client)

    collected = [event async for event in model.astream([Message.user("hi")])]

    assert [e for e in collected if isinstance(e, TextDelta)][0].text == "Hi"
    assert isinstance(collected[-1], MessageStop)
    assert collected[-1].response.text == "Hi async"
