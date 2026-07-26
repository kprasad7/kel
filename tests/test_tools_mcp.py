import pytest

from kel.tools.mcp_tool import MCPToolset


class FakeMCPTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class FakeSession:
    """Mimics the sync-shaped surface MCPToolset needs, bypassing the
    real async ClientSession/background-thread machinery entirely — same
    DI pattern (session=) as the provider adapters' client=."""

    def __init__(self, tools, call_results):
        self._tools = tools
        self._call_results = call_results
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return self._tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._call_results[name]


def test_as_kel_tools_translates_mcp_tool_schema_to_kel_tool():
    session = FakeSession(
        tools=[FakeMCPTool("get_weather", "get current weather", {"type": "object", "properties": {"city": {}}})],
        call_results={},
    )
    toolset = MCPToolset(session=session)

    tools = toolset.as_kel_tools()

    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].description == "get current weather"
    assert tools[0].input_schema == {"type": "object", "properties": {"city": {}}}


def test_kel_tool_call_delegates_to_the_mcp_session():
    session = FakeSession(
        tools=[FakeMCPTool("get_weather", "get weather", {"type": "object"})],
        call_results={"get_weather": "sunny in Paris"},
    )
    toolset = MCPToolset(session=session)
    tools = toolset.as_kel_tools()

    result = tools[0]({"city": "Paris"})

    assert result == "sunny in Paris"
    assert session.calls == [("get_weather", {"city": "Paris"})]


def test_as_kel_tools_handles_missing_description_and_schema():
    session = FakeSession(tools=[FakeMCPTool("noop", None, None)], call_results={})
    toolset = MCPToolset(session=session)

    tools = toolset.as_kel_tools()

    assert tools[0].description == ""
    assert tools[0].input_schema == {"type": "object"}


def test_multiple_mcp_tools_translate_independently():
    session = FakeSession(
        tools=[
            FakeMCPTool("tool_a", "does a", {"type": "object"}),
            FakeMCPTool("tool_b", "does b", {"type": "object"}),
        ],
        call_results={"tool_a": "result a", "tool_b": "result b"},
    )
    toolset = MCPToolset(session=session)

    tools = {t.name: t for t in toolset.as_kel_tools()}

    assert tools["tool_a"]({}) == "result a"
    assert tools["tool_b"]({}) == "result b"


def test_close_is_a_no_op_for_an_injected_session():
    # closing shouldn't try to tear down a background thread/loop that
    # was never created when the connection wasn't owned by this toolset
    session = FakeSession(tools=[], call_results={})
    toolset = MCPToolset(session=session)

    toolset.close()  # must not raise


def test_requires_either_server_params_or_session():
    with pytest.raises(ValueError):
        MCPToolset()


def test_context_manager_calls_close():
    session = FakeSession(tools=[], call_results={})
    with MCPToolset(session=session) as toolset:
        assert toolset.list_tools() == []
