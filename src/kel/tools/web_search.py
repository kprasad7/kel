"""Built-in web search tools — the kel equivalent of LangChain's
`DuckDuckGoSearchRun` / `WikipediaQueryRun` / `TavilySearchResults` /
`GoogleSearchAPIWrapper` / `BingSearchAPIWrapper`. All return a
`kel.agents.Tool`, ready to hand straight to `Agent(..., tools=[...])`.

No-key options:
- `wikipedia_search_tool`: zero dependency, uses Wikipedia's official
  search API (`en.wikipedia.org/w/api.php`) — a real JSON API, not a
  scrape. Good general-knowledge coverage; snippets are short and
  truncated, not reliable alone for specific facts (pair with
  `kel.tools.fetch_url_tool`). Not a live news search.
- `duckduckgo_search_tool`: scrapes DuckDuckGo's HTML lite interface.
  **Known broken as of testing this (2026-07)**: DDG now serves an
  anti-bot CAPTCHA challenge page to non-browser requests, so this
  reliably returns "no results found" — kept for reference only.

API-key options — all real, official, ToS-compliant search APIs (not
scrapes), so pick whichever one you already have a key for:
- `tavily_search_tool` (https://tavily.com, generous free tier):
  purpose-built for AI agents, results come back agent-ready, covers
  current events. The best default if you don't already have one of the below.
- `brave_search_tool` (https://brave.com/search/api/, free tier available)
- `serpapi_search_tool` (https://serpapi.com, wraps real Google results legitimately)
- `bing_search_tool` (Azure Bing Web Search, https://portal.azure.com)
- `google_custom_search_tool` (Google's official Custom Search JSON API,
  https://programmablesearchengine.google.com — needs both an API key and a
  Search Engine ID ("cx"), free tier: 100 queries/day)

Use `kel.tools.get_web_search_tool(provider, api_key=..., **kwargs)` to
pick a provider dynamically at runtime instead of importing a specific
factory — the same "provider:model_id" pattern as `kel.get_model`.

All fetch/post functions are injectable so tests never hit the network —
same pattern as the provider adapters' injectable `client`.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable

from kel.agents.tool import Tool

_WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "the search query"}},
    "required": ["query"],
}

# --------------------------------------------------------------------------
# DuckDuckGo (no API key)
# --------------------------------------------------------------------------

_RESULT_LINK_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


def _decode_ddg_redirect(href: str) -> str:
    """DDG's html interface wraps outbound links in a redirect
    (`//duckduckgo.com/l/?uddg=<url-encoded-target>&rut=...`) — unwrap it
    so the agent gets the real destination URL, not a DDG redirect link."""
    parsed = urllib.parse.urlparse(href)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query:
        return urllib.parse.unquote(query["uddg"][0])
    return href if href.startswith("http") else f"https:{href}"


def parse_ddg_html(html: str, max_results: int) -> list[dict]:
    links = _RESULT_LINK_RE.findall(html)
    snippets = _SNIPPET_RE.findall(html)
    results = []
    for i, (href, title) in enumerate(links[:max_results]):
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        results.append({"url": _decode_ddg_redirect(href), "title": _strip_tags(title), "snippet": snippet})
    return results


def _default_ddg_fetch(query: str) -> str:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; kel-agent/0.1)"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read().decode("utf-8", errors="ignore")


def _format_results(results: list[dict]) -> str:
    if not results:
        return "no results found"
    return "\n\n".join(
        f"{r['title']}\n{r['url']}\n{r.get('snippet', '')}".strip() for r in results
    )


def duckduckgo_search_tool(*, max_results: int = 5, fetch: Callable[[str], str] | None = None) -> Tool:
    fetch = fetch or _default_ddg_fetch

    def run(tool_input: dict) -> str:
        html = fetch(tool_input["query"])
        return _format_results(parse_ddg_html(html, max_results))

    return Tool(
        name="web_search",
        description="Search the web via DuckDuckGo and return the top results (title, url, snippet).",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# Wikipedia (no API key, real JSON API — not a scrape)
# --------------------------------------------------------------------------

_SEARCHMATCH_RE = re.compile(r'<span class="searchmatch">(.*?)</span>')


def _default_wikipedia_fetch(query: str, max_results: int) -> dict:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"}
    )
    # Wikipedia's API etiquette asks for a descriptive User-Agent identifying the app
    req = urllib.request.Request(url, headers={"User-Agent": "kel-agent/0.1 (https://github.com/kprasad7/kel)"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return json.loads(resp.read())


def wikipedia_search_tool(
    *, max_results: int = 3, fetch: Callable[[str, int], dict] | None = None
) -> Tool:
    fetch = fetch or _default_wikipedia_fetch

    def run(tool_input: dict) -> str:
        data = fetch(tool_input["query"], max_results)
        hits = data.get("query", {}).get("search", [])
        results = [
            {
                "title": hit.get("title", ""),
                "url": f"https://en.wikipedia.org/wiki/{hit.get('title', '').replace(' ', '_')}",
                "snippet": _strip_tags(_SEARCHMATCH_RE.sub(r"\1", hit.get("snippet", ""))),
            }
            for hit in hits
        ]
        return _format_results(results)

    return Tool(
        name="wikipedia_search",
        description="Search Wikipedia for factual/background/encyclopedic information.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# Tavily (API key required, agent-oriented)
# --------------------------------------------------------------------------


def _default_tavily_post(payload: bytes) -> bytes:
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read()


def tavily_search_tool(
    api_key: str, *, max_results: int = 5, post: Callable[[bytes], bytes] | None = None
) -> Tool:
    post = post or _default_tavily_post

    def run(tool_input: dict) -> str:
        payload = json.dumps(
            {"api_key": api_key, "query": tool_input["query"], "max_results": max_results}
        ).encode("utf-8")
        data = json.loads(post(payload))
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:300]}
            for r in data.get("results", [])
        ]
        return _format_results(results)

    return Tool(
        name="web_search",
        description="Search the web via Tavily (AI-agent-oriented search API) and return the top results.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# Brave Search (API key required)
# --------------------------------------------------------------------------


def _default_brave_fetch(query: str, api_key: str, max_results: int) -> bytes:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results}
    )
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "X-Subscription-Token": api_key}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read()


def brave_search_tool(
    api_key: str, *, max_results: int = 5, fetch: Callable[[str, str, int], bytes] | None = None
) -> Tool:
    fetch = fetch or _default_brave_fetch

    def run(tool_input: dict) -> str:
        data = json.loads(fetch(tool_input["query"], api_key, max_results))
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
            for r in data.get("web", {}).get("results", [])
        ]
        return _format_results(results)

    return Tool(
        name="web_search",
        description="Search the web via Brave Search and return the top results.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# SerpAPI (API key required — wraps real Google results legitimately)
# --------------------------------------------------------------------------


def _default_serpapi_fetch(query: str, api_key: str, max_results: int) -> bytes:
    url = "https://serpapi.com/search?" + urllib.parse.urlencode(
        {"q": query, "api_key": api_key, "engine": "google", "num": max_results}
    )
    with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read()


def serpapi_search_tool(
    api_key: str, *, max_results: int = 5, fetch: Callable[[str, str, int], bytes] | None = None
) -> Tool:
    fetch = fetch or _default_serpapi_fetch

    def run(tool_input: dict) -> str:
        data = json.loads(fetch(tool_input["query"], api_key, max_results))
        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("organic_results", [])[:max_results]
        ]
        return _format_results(results)

    return Tool(
        name="web_search",
        description="Search the web via SerpAPI (real Google results, official API) and return the top results.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# Bing Web Search (Azure, API key required)
# --------------------------------------------------------------------------


def _default_bing_fetch(query: str, api_key: str, max_results: int) -> bytes:
    url = "https://api.bing.microsoft.com/v7.0/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results}
    )
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": api_key})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read()


def bing_search_tool(
    api_key: str, *, max_results: int = 5, fetch: Callable[[str, str, int], bytes] | None = None
) -> Tool:
    fetch = fetch or _default_bing_fetch

    def run(tool_input: dict) -> str:
        data = json.loads(fetch(tool_input["query"], api_key, max_results))
        results = [
            {"title": r.get("name", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
            for r in data.get("webPages", {}).get("value", [])
        ]
        return _format_results(results)

    return Tool(
        name="web_search",
        description="Search the web via Bing Web Search (Azure) and return the top results.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )


# --------------------------------------------------------------------------
# Google Custom Search JSON API (API key + search-engine ID required)
# --------------------------------------------------------------------------


def _default_google_fetch(query: str, api_key: str, cx: str, max_results: int) -> bytes:
    url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(
        {"q": query, "key": api_key, "cx": cx, "num": min(max_results, 10)}
    )
    with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - hardcoded kel-controlled hostname, only query params vary
        return resp.read()


def google_custom_search_tool(
    api_key: str,
    cx: str,
    *,
    max_results: int = 5,
    fetch: Callable[[str, str, str, int], bytes] | None = None,
) -> Tool:
    fetch = fetch or _default_google_fetch

    def run(tool_input: dict) -> str:
        data = json.loads(fetch(tool_input["query"], api_key, cx, max_results))
        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("items", [])
        ]
        return _format_results(results)

    return Tool(
        name="web_search",
        description="Search the web via Google Custom Search (official API) and return the top results.",
        input_schema=_WEB_SEARCH_SCHEMA,
        fn=run,
    )
