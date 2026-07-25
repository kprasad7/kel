import json

from kel.tools import (
    bing_search_tool,
    brave_search_tool,
    duckduckgo_search_tool,
    google_custom_search_tool,
    parse_ddg_html,
    serpapi_search_tool,
    tavily_search_tool,
    wikipedia_search_tool,
)

_SAMPLE_DDG_HTML = """
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fkel&amp;rut=abc">
        kel &mdash; universal agentic OS
      </a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fkel">
      An open source <b>agentic</b> framework with observability built in.
    </a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2Fpage&amp;rut=def">
        Other Result
      </a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2Fpage">
      A second, unrelated result.
    </a>
  </div>
</div>
"""


def test_parse_ddg_html_extracts_title_url_and_snippet():
    results = parse_ddg_html(_SAMPLE_DDG_HTML, max_results=5)

    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/kel"
    assert "kel" in results[0]["title"]
    assert "agentic" in results[0]["snippet"]
    assert results[1]["url"] == "https://other.com/page"


def test_parse_ddg_html_respects_max_results():
    results = parse_ddg_html(_SAMPLE_DDG_HTML, max_results=1)
    assert len(results) == 1


def test_parse_ddg_html_returns_empty_list_for_no_matches():
    assert parse_ddg_html("<html><body>no results here</body></html>", max_results=5) == []


def test_duckduckgo_search_tool_wires_fetch_and_formats_results():
    captured_queries = []

    def fake_fetch(query: str) -> str:
        captured_queries.append(query)
        return _SAMPLE_DDG_HTML

    tool = duckduckgo_search_tool(max_results=5, fetch=fake_fetch)

    assert tool.name == "web_search"
    result_text = tool({"query": "what is kel"})

    assert captured_queries == ["what is kel"]
    assert "example.com/kel" in result_text
    assert "other.com/page" in result_text


def test_duckduckgo_search_tool_handles_no_results():
    tool = duckduckgo_search_tool(fetch=lambda query: "<html></html>")
    assert tool({"query": "nothing"}) == "no results found"


def test_tavily_search_tool_wires_post_and_sends_query_and_api_key():
    captured_payloads = []

    def fake_post(payload: bytes) -> bytes:
        captured_payloads.append(json.loads(payload))
        return json.dumps(
            {
                "results": [
                    {"title": "kel repo", "url": "https://github.com/kprasad7/kel", "content": "A universal agentic OS."}
                ]
            }
        ).encode("utf-8")

    tool = tavily_search_tool("tvly-secret-key", max_results=3, post=fake_post)
    result_text = tool({"query": "kel agentic os"})

    assert captured_payloads[0] == {"api_key": "tvly-secret-key", "query": "kel agentic os", "max_results": 3}
    assert "github.com/kprasad7/kel" in result_text
    assert "kel repo" in result_text


def test_tavily_search_tool_handles_no_results():
    tool = tavily_search_tool("key", post=lambda payload: json.dumps({"results": []}).encode("utf-8"))
    assert tool({"query": "nothing"}) == "no results found"


def test_wikipedia_search_tool_wires_fetch_and_formats_results():
    captured = []

    def fake_fetch(query: str, max_results: int) -> dict:
        captured.append((query, max_results))
        return {
            "query": {
                "search": [
                    {
                        "title": "Super Bowl",
                        "snippet": 'The <span class="searchmatch">Super</span> Bowl is the annual championship game.',
                    }
                ]
            }
        }

    tool = wikipedia_search_tool(max_results=3, fetch=fake_fetch)

    assert tool.name == "wikipedia_search"
    result_text = tool({"query": "super bowl"})

    assert captured == [("super bowl", 3)]
    assert "Super Bowl" in result_text
    assert "en.wikipedia.org/wiki/Super_Bowl" in result_text
    assert "<span" not in result_text  # searchmatch markup stripped, plain text kept
    assert "championship game" in result_text


def test_wikipedia_search_tool_handles_no_results():
    tool = wikipedia_search_tool(fetch=lambda query, max_results: {"query": {"search": []}})
    assert tool({"query": "asdkjfhaskjdfh"}) == "no results found"


def test_brave_search_tool_wires_key_and_formats_results():
    captured = []

    def fake_fetch(query, api_key, max_results):
        captured.append((query, api_key, max_results))
        return json.dumps(
            {"web": {"results": [{"title": "kel", "url": "https://example.com/kel", "description": "an agentic OS"}]}}
        ).encode("utf-8")

    tool = brave_search_tool("brave-key", max_results=4, fetch=fake_fetch)
    result = tool({"query": "kel agentic os"})

    assert captured == [("kel agentic os", "brave-key", 4)]
    assert "example.com/kel" in result
    assert "an agentic OS" in result


def test_serpapi_search_tool_wires_key_and_formats_results():
    captured = []

    def fake_fetch(query, api_key, max_results):
        captured.append((query, api_key, max_results))
        return json.dumps(
            {"organic_results": [{"title": "kel repo", "link": "https://github.com/kprasad7/kel", "snippet": "universal agentic OS"}]}
        ).encode("utf-8")

    tool = serpapi_search_tool("serp-key", max_results=3, fetch=fake_fetch)
    result = tool({"query": "kel github"})

    assert captured == [("kel github", "serp-key", 3)]
    assert "github.com/kprasad7/kel" in result


def test_bing_search_tool_wires_key_and_formats_results():
    captured = []

    def fake_fetch(query, api_key, max_results):
        captured.append((query, api_key, max_results))
        return json.dumps(
            {"webPages": {"value": [{"name": "kel", "url": "https://example.com/kel", "snippet": "agentic OS"}]}}
        ).encode("utf-8")

    tool = bing_search_tool("bing-key", fetch=fake_fetch)
    result = tool({"query": "kel"})

    assert captured[0][1] == "bing-key"
    assert "example.com/kel" in result


def test_google_custom_search_tool_wires_key_cx_and_formats_results():
    captured = []

    def fake_fetch(query, api_key, cx, max_results):
        captured.append((query, api_key, cx, max_results))
        return json.dumps(
            {"items": [{"title": "kel", "link": "https://example.com/kel", "snippet": "agentic OS"}]}
        ).encode("utf-8")

    tool = google_custom_search_tool("google-key", "cx-123", max_results=2, fetch=fake_fetch)
    result = tool({"query": "kel"})

    assert captured == [("kel", "google-key", "cx-123", 2)]
    assert "example.com/kel" in result


def test_all_keyed_search_tools_handle_no_results():
    assert brave_search_tool("k", fetch=lambda q, a, m: b'{"web": {"results": []}}')({"query": "x"}) == "no results found"
    assert serpapi_search_tool("k", fetch=lambda q, a, m: b'{"organic_results": []}')({"query": "x"}) == "no results found"
    assert bing_search_tool("k", fetch=lambda q, a, m: b'{"webPages": {"value": []}}')({"query": "x"}) == "no results found"
    assert google_custom_search_tool("k", "cx", fetch=lambda q, a, c, m: b'{"items": []}')({"query": "x"}) == "no results found"
