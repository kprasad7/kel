from __future__ import annotations

from pathlib import Path

from kel.specs.frontmatter import split_frontmatter
from kel.specs.types import AgentSpec

_KNOWN_KEYS = {"name", "model", "api_key_ref", "tools", "budget", "version"}


def parse_agent_spec(text: str, *, default_name: str = "agent") -> AgentSpec:
    data, body = split_frontmatter(text)
    if "model" not in data:
        raise ValueError("agent spec frontmatter must include 'model' (e.g. 'anthropic:claude-sonnet-5')")
    return AgentSpec(
        name=data.get("name", default_name),
        model=data["model"],
        system_prompt=body.strip(),
        api_key_ref=data.get("api_key_ref"),
        tools=data.get("tools", []),
        budget=data.get("budget"),
        version=data.get("version"),
        metadata={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
    )


def load_agent_spec(path: str | Path) -> AgentSpec:
    path = Path(path)
    return parse_agent_spec(path.read_text(encoding="utf-8"), default_name=path.stem)


class AgentSpecLoader:
    """Hot-reloads a spec file when its mtime changes (DESIGN.md 3.6:
    "hot-reloadable"), so editing a running agent's `.md` file takes
    effect on the next `load()` without restarting the process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: float | None = None
        self._cached: AgentSpec | None = None

    def load(self) -> AgentSpec:
        mtime = self.path.stat().st_mtime
        if self._cached is None or mtime != self._mtime:
            self._cached = load_agent_spec(self.path)
            self._mtime = mtime
        return self._cached
