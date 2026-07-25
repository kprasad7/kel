"""Generic URL content fetcher. Dynamic in the sense that matters: it
extracts readable text from *any* page's HTML structure via a generic
tag-based parser (skip script/style/nav/header/footer, keep the rest),
not a regex tied to one specific site's CSS class names the way a
search-engine-result scraper has to be.

This fetches whatever URL it's given as a normal HTTP client — it is not
an anti-bot workaround. Most ordinary web pages (articles, docs, wikis)
don't have bot walls; the small subset that actively block automated
access (major search engines' result pages, some paywalled sites) will
return an error here rather than fabricated content, same as any honest
HTTP client would get.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser

from kel.agents.tool import Tool

_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "svg"}
_ALLOWED_SCHEMES = {"http", "https"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)


def extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()


def _default_url_fetch(url: str) -> str:
    # Flagged by Bandit (B310) for good reason: urllib.urlopen on an
    # unvalidated URL can be pointed at file:// (local file disclosure) or
    # other exotic schemes it supports — a real SSRF-adjacent risk for a
    # tool whose whole job is fetching an agent/user-supplied URL. Restrict
    # to http/https before ever calling urlopen.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported URL scheme {scheme!r} — only http/https are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "kel-agent/0.1 (+https://github.com/kprasad7/kel)"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - scheme validated above
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def fetch_url_tool(*, max_chars: int = 4000, fetch: Callable[[str], str] | None = None) -> Tool:
    fetch = fetch or _default_url_fetch

    def run(tool_input: dict) -> str:
        url = tool_input["url"]
        try:
            html = fetch(url)
        except Exception as exc:
            return f"error fetching {url}: {exc}"
        text = extract_text(html)
        if not text:
            return f"no readable text content found at {url}"
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return text

    return Tool(
        name="fetch_url",
        description=(
            "Fetch a specific URL (e.g. one returned by wikipedia_search or given by the "
            "user) and return its readable text content, stripped of scripts/nav/ads markup."
        ),
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "the URL to fetch"}},
            "required": ["url"],
        },
        fn=run,
    )
