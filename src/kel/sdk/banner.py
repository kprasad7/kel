"""ASCII banner shown by `kel --version` and bare `kel` invocations.

Not shown during `pip install` itself — a wheel install can't run
arbitrary code by design (that's a security feature, not a gap), so the
practical equivalent is showing it the first time a user actually
invokes the `kel` command after installing.
"""

from __future__ import annotations

import os
import sys

from kel import __version__

_ART = r"""
  _              _
 | | _____ _ __ | |
 | |/ / _ \ '_ \| |
 |   <  __/ | | | |
 |_|\_\___|_| |_|_|
"""

_TAGLINE = "Universal agentic OS for building production-grade AI agents"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def render_banner() -> str:
    if _supports_color():
        cyan, dim, reset = "\033[36m", "\033[2m", "\033[0m"
    else:
        cyan = dim = reset = ""
    lines = [f"{cyan}{_ART}{reset}"]
    lines.append(f"{dim}{_TAGLINE} - v{__version__}{reset}")
    return "\n".join(lines)


def print_banner() -> None:
    print(render_banner())
