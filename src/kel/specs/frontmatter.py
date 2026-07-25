"""Splits a `.md` file into its YAML frontmatter (between `---` fences)
and body. Deliberately minimal — not a full Markdown parser, just what's
needed to treat `.md` files as structured config (DESIGN.md principle 4)."""

from __future__ import annotations

import yaml


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw_frontmatter = text[4:end]
    body = text[end + 4 :].lstrip("\n").lstrip("\r\n")
    data = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return data, body
