"""Shared result type for `kel.media`. Every fal.ai model endpoint (image,
video, audio, lipsync) returns its own response shape — kel does not
hardcode one — so `MediaResult` keeps the full raw response and adds a
best-effort `urls` extraction (every string found under any "url" key,
recursively) that works across the different endpoint schemas without
kel needing to special-case each one."""

from __future__ import annotations

from typing import Any


def _extract_urls(node: Any, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "url" and isinstance(value, str):
                found.append(value)
            else:
                _extract_urls(value, found)
    elif isinstance(node, list):
        for item in node:
            _extract_urls(item, found)


class MediaResult:
    """`raw` is the full, endpoint-specific response dict — the source of
    truth. `urls` is a convenience: every string value found under a
    "url" key anywhere in `raw`, in the order encountered."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        urls: list[str] = []
        _extract_urls(raw, urls)
        self.urls = urls
