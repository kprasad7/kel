from kel.tools import extract_text, fetch_url_tool

_ARTICLE_HTML = """
<html><head><title>Article</title><style>body{color:red}</style></head>
<body>
  <nav>Home | About | Contact</nav>
  <header><h1>Site Name</h1></header>
  <article>
    <h2>kel is a universal agentic OS</h2>
    <p>It ships observability, budget control, and multi-agent orchestration.</p>
  </article>
  <script>console.log('tracking pixel');</script>
  <footer>Copyright 2026</footer>
</body></html>
"""

_DIFFERENTLY_STRUCTURED_HTML = """
<!DOCTYPE html>
<div id="wrapper">
  <div class="whatever-random-class-name-12345">
    <span>Completely different markup structure</span>
    <p>But the extractor should still pull this text out fine.</p>
  </div>
</div>
"""


def test_extract_text_strips_nav_header_footer_script_and_style():
    text = extract_text(_ARTICLE_HTML)

    assert "kel is a universal agentic OS" in text
    assert "observability, budget control" in text
    assert "Home | About | Contact" not in text
    assert "Copyright 2026" not in text
    assert "tracking pixel" not in text
    assert "color:red" not in text


def test_extract_text_works_on_arbitrary_unrelated_markup_structure():
    # proves the extractor is generic (tag-based), not tied to any one
    # site's specific CSS classes the way the DDG-result scraper was
    text = extract_text(_DIFFERENTLY_STRUCTURED_HTML)
    assert "Completely different markup structure" in text
    assert "should still pull this text out fine" in text


def test_extract_text_collapses_whitespace():
    html = "<p>hello    \n\n   world</p>"
    assert extract_text(html) == "hello world"


def test_fetch_url_tool_wires_fetch_and_extracts_text():
    captured_urls = []

    def fake_fetch(url: str) -> str:
        captured_urls.append(url)
        return _ARTICLE_HTML

    tool = fetch_url_tool(fetch=fake_fetch)
    result = tool({"url": "https://example.com/kel"})

    assert tool.name == "fetch_url"
    assert captured_urls == ["https://example.com/kel"]
    assert "universal agentic OS" in result


def test_fetch_url_tool_truncates_long_content():
    def fake_fetch(url: str) -> str:
        return "<p>" + ("word " * 2000) + "</p>"

    tool = fetch_url_tool(max_chars=100, fetch=fake_fetch)
    result = tool({"url": "https://example.com/long"})

    assert len(result) <= 120
    assert result.endswith("...[truncated]")


def test_fetch_url_tool_handles_fetch_error_gracefully():
    def failing_fetch(url: str) -> str:
        raise TimeoutError("connection timed out")

    tool = fetch_url_tool(fetch=failing_fetch)
    result = tool({"url": "https://example.com/down"})

    assert "error fetching" in result
    assert "connection timed out" in result


def test_fetch_url_tool_handles_empty_page():
    tool = fetch_url_tool(fetch=lambda url: "<html><body></body></html>")
    result = tool({"url": "https://example.com/empty"})
    assert "no readable text content" in result


def test_default_url_fetch_rejects_file_scheme():
    from kel.tools.web_fetch import _default_url_fetch

    try:
        _default_url_fetch("file:///etc/passwd")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unsupported URL scheme" in str(exc)


def test_default_url_fetch_rejects_ftp_scheme():
    from kel.tools.web_fetch import _default_url_fetch

    try:
        _default_url_fetch("ftp://example.com/file")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unsupported URL scheme" in str(exc)


def test_fetch_url_tool_surfaces_disallowed_scheme_as_a_graceful_error():
    # uses the REAL default fetch (no injected fake) to prove the tool as
    # a whole refuses file:// rather than silently reading a local file
    tool = fetch_url_tool()
    result = tool({"url": "file:///etc/passwd"})
    assert "error fetching" in result
    assert "unsupported URL scheme" in result
