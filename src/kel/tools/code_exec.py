"""Python code execution tool, deliberately built to be honest about what
it isn't.

**This is process isolation with a timeout, NOT a security sandbox.**
Executed code runs in a separate Python subprocess (so it can't corrupt
the agent process's own memory/state, and a hung/infinite-looping script
gets killed by the timeout instead of hanging the agent forever) — but it
has the same filesystem, network, and OS access as the account running
kel. It does not stop code from reading your files, making network calls,
or spawning further processes. Appropriate for trusted code generation
(your own agent writing its own scratch code); NOT appropriate for running
code from an untrusted or adversarial source. For that, use a real
sandboxed runtime (a container, gVisor, or a hosted sandbox like E2B/
Modal) — kel does not ship one, and pretending a subprocess is equivalent
to one would be actively misleading.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

from kel.agents.tool import Tool

RunSubprocess = Callable[[list[str], float], subprocess.CompletedProcess]


def _default_run_subprocess(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def python_exec_tool(*, timeout: float = 5.0, run_subprocess: RunSubprocess | None = None) -> Tool:
    run_subprocess = run_subprocess or _default_run_subprocess

    def run(tool_input: dict) -> str:
        code = tool_input["code"]
        try:
            result = run_subprocess([sys.executable, "-c", code], timeout)
        except subprocess.TimeoutExpired:
            return f"error: code execution timed out after {timeout}s"

        output = result.stdout or ""
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]\n{result.stderr}"
        return output.strip() or "(no output)"

    return Tool(
        name="python_exec",
        description=(
            "Execute Python code in a subprocess and return its stdout/stderr. "
            "Process-isolated with a timeout, but NOT a security sandbox — has full "
            "filesystem/network access. Only for trusted code generation."
        ),
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "the Python code to run"}},
            "required": ["code"],
        },
        fn=run,
    )
