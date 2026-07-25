"""Dual-path response (DESIGN.md 3.12): a fast path answers directly from
context/working memory while a slow path (RAG query, tool call) runs
concurrently, so the agent can emit a filler ("let me check that") instead
of going silent during the round trip. This is the actual orchestration
value kel.realtime provides — everything audio/video is left to real
vendor SDKs behind the Protocols in providers.py."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


def run_dual_path(
    fast: Callable[[], str | None],
    slow: Callable[[], str],
    *,
    on_filler: Callable[[str], None] | None = None,
) -> str:
    """Starts `slow` in the background immediately, then calls `fast`
    synchronously (it should be near-instant — e.g. reading working memory,
    not a network call). If `fast` returns non-None text, `on_filler` is
    invoked with it right away (e.g. to start TTS speaking the filler)
    while `slow` keeps running. Returns `slow`'s result once it completes.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        slow_future = pool.submit(slow)
        filler = fast()
        if filler is not None and on_filler is not None:
            on_filler(filler)
        return slow_future.result()
