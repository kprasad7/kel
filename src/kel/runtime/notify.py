"""Generic notification dispatch for durable human-in-the-loop. kel's
`Interrupt`/`resume_graph()`/`CheckpointStore` already let a run pause
and resume arbitrarily later — an interrupted `GraphRun` (or any
`Checkpoint`) is just data, persist it anywhere, including for weeks.
What was missing was telling a human it's actually waiting on them.

`Notifier` is the same injectable-Protocol shape as
`EpisodicStore`/`CheckpointStore` — plug in your own Slack/email/SMS
sender behind it. `WebhookNotifier` is the one concrete,
zero-new-required-dependency implementation shipped here (a plain HTTP
POST via stdlib `urllib`) — Slack incoming webhooks, PagerDuty, and most
custom "send an email/SMS" endpoints all accept exactly this shape, so it
covers the common case without a vendor SDK per notification channel.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from kel.runtime.executor import GraphRun

_ALLOWED_SCHEMES = {"http", "https"}


class Notifier(Protocol):
    def notify(self, message: str, *, metadata: dict[str, Any] | None = None) -> None: ...


def _default_post(url: str, payload: bytes) -> None:
    # same SSRF-hardening as kel.tools.web_fetch: validate the scheme
    # before ever calling urlopen, since this URL is often configuration
    # a caller controls but shouldn't be able to point at file:// etc.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported webhook URL scheme {scheme!r} — only http/https are allowed")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as _resp:  # nosec B310 - scheme validated above
        pass


class WebhookNotifier:
    """POSTs `{"message": ..., "metadata": ...}` as JSON to `url`. Works
    as-is against Slack incoming webhooks (which read `message` as the
    top-level `text` field if you set `url` to a webhook that expects
    that shape — otherwise wrap `post=` to reshape the payload for
    whatever your endpoint expects)."""

    def __init__(self, url: str, *, post: Callable[[str, bytes], None] | None = None):
        self.url = url
        self._post = post or _default_post

    def notify(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        payload = json.dumps({"message": message, "metadata": metadata or {}}).encode("utf-8")
        self._post(self.url, payload)


def notify_interrupt(run: GraphRun, notifier: Notifier, *, message: str | None = None) -> None:
    """Convenience: notify a human that a graph run is paused waiting on
    them, using the interrupted `GraphRun`'s own identifying info. Call
    this right after `run_graph()`/`resume_graph()` returns with
    `run.interrupted` true — persist `run` (or a `Checkpoint`) somewhere
    durable, dispatch this notification, and call `resume_graph()`
    whenever the actual response comes back (minutes, hours, or weeks
    later — nothing here assumes it happens quickly)."""
    if not run.interrupted:
        raise ValueError("this GraphRun is not interrupted; nothing to notify about")
    text = message or f"Run {run.run_id} is paused at node {run.pending_node!r}, waiting for input."
    notifier.notify(
        text,
        metadata={"run_id": run.run_id, "pending_node": run.pending_node, "payload": run.interrupt_payload},
    )
