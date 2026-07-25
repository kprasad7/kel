from kel import Message, ModelResponse, Role, TextPart, ToolUsePart, Usage


def test_message_user_helper_roundtrips_text():
    msg = Message.user("hello")
    assert msg.role == Role.USER
    assert msg.text == "hello"


def test_message_text_concatenates_only_text_parts():
    msg = Message(
        role=Role.ASSISTANT,
        content=[TextPart(text="a"), ToolUsePart(id="1", name="x", input={}), TextPart(text="b")],
    )
    assert msg.text == "ab"


def test_usage_total_tokens():
    usage = Usage(input_tokens=10, output_tokens=5)
    assert usage.total_tokens == 15


def test_model_response_text_and_tool_calls():
    tool_call = ToolUsePart(id="1", name="search", input={"q": "x"})
    resp = ModelResponse(
        id="r1",
        model="m1",
        content=[TextPart(text="hi"), tool_call],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    assert resp.text == "hi"
    assert resp.tool_calls == [tool_call]
