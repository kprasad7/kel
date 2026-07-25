import pytest

from kel.tools import get_web_search_tool, register_search_provider


def test_get_web_search_tool_resolves_no_key_provider():
    tool = get_web_search_tool(
        "wikipedia", fetch=lambda query, max_results: {"query": {"search": []}}
    )
    assert tool.name == "wikipedia_search"


def test_get_web_search_tool_resolves_keyed_provider():
    tool = get_web_search_tool(
        "tavily", api_key="tvly-x", post=lambda payload: b'{"results": []}'
    )
    assert tool.name == "web_search"
    assert tool({"query": "x"}) == "no results found"


def test_get_web_search_tool_resolves_google_with_cx():
    tool = get_web_search_tool(
        "google", api_key="k", cx="cx-1", fetch=lambda q, a, c, m: b'{"items": []}'
    )
    assert tool({"query": "x"}) == "no results found"


def test_get_web_search_tool_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown search provider"):
        get_web_search_tool("not-a-real-provider")


def test_register_search_provider_allows_custom_providers():
    from kel.agents.tool import Tool

    def custom_factory(**kwargs):
        return Tool(name="custom_search", description="d", input_schema={}, fn=lambda i: "custom result")

    register_search_provider("my-custom-engine", custom_factory)
    tool = get_web_search_tool("my-custom-engine")
    assert tool.name == "custom_search"
    assert tool({}) == "custom result"


def test_all_builtin_providers_are_registered():
    from kel.tools.search_registry import _SEARCH_PROVIDERS

    for name in ["wikipedia", "duckduckgo", "tavily", "brave", "serpapi", "bing", "google"]:
        assert name in _SEARCH_PROVIDERS
