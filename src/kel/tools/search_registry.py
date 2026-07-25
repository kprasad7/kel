"""Dynamic web-search provider selection — the same pattern as
`kel.models.get_model("provider:model_id")`, so an application can pick
whichever search provider its user actually has a key for at runtime,
instead of hardcoding one factory import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kel.agents.tool import Tool

_ToolFactory = Callable[..., Tool]

_SEARCH_PROVIDERS: dict[str, _ToolFactory] = {}


def register_search_provider(name: str, factory: _ToolFactory) -> None:
    """Register a `Tool`-returning factory under a provider name.
    `factory` is called as `factory(**kwargs)` from `get_web_search_tool`."""
    _SEARCH_PROVIDERS[name] = factory


def get_web_search_tool(provider: str, **kwargs: Any) -> Tool:
    """Build a web-search Tool for the named provider.

    Examples:
        get_web_search_tool("wikipedia", max_results=3)
        get_web_search_tool("tavily", api_key="tvly-...", max_results=5)
        get_web_search_tool("google", api_key="...", cx="...", max_results=5)

    Known providers: wikipedia, duckduckgo (no key); tavily, brave,
    serpapi, bing, google (key required — google also needs `cx`).
    """
    factory = _SEARCH_PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(f"Unknown search provider {provider!r}. Known providers: {sorted(_SEARCH_PROVIDERS)}")
    return factory(**kwargs)


def _register_builtins() -> None:
    from kel.tools.web_search import (
        bing_search_tool,
        brave_search_tool,
        duckduckgo_search_tool,
        google_custom_search_tool,
        serpapi_search_tool,
        tavily_search_tool,
        wikipedia_search_tool,
    )

    register_search_provider("wikipedia", wikipedia_search_tool)
    register_search_provider("duckduckgo", duckduckgo_search_tool)
    register_search_provider("tavily", tavily_search_tool)
    register_search_provider("brave", brave_search_tool)
    register_search_provider("serpapi", serpapi_search_tool)
    register_search_provider("bing", bing_search_tool)
    register_search_provider("google", google_custom_search_tool)


_register_builtins()
