"""Shell command execution tool. Same honest framing as `python_exec_tool`
— process-isolated with a timeout, **not a security sandbox** — plus an
extra risk `python_exec_tool` doesn't have: this runs through a real
shell, so pipes, redirects, and shell metacharacters all work exactly as
they would in a terminal, including shell-injection-style risk if any part
of the command string comes from untrusted input. Only for trusted code
generation (your own agent writing its own commands), never for running
a command built from user-supplied text without validating it yourself.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from kel.agents.tool import Tool

RunSubprocess = Callable[[str, float], subprocess.CompletedProcess]


def _default_run_subprocess(command: str, timeout: float) -> subprocess.CompletedProcess:
    # nosec B602 - shell=True is the deliberate point of this tool (running
    # a shell command string, pipes/redirects included), not an oversight;
    # the risk this flags is exactly what the module docstring above warns
    # callers about — this is a documented capability, not a fixable bug.
    return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)  # nosec B602


def shell_exec_tool(*, timeout: float = 5.0, run_subprocess: RunSubprocess | None = None) -> Tool:
    run_subprocess = run_subprocess or _default_run_subprocess

    def run(tool_input: dict) -> str:
        command = tool_input["command"]
        try:
            result = run_subprocess(command, timeout)
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"

        output = result.stdout or ""
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]\n{result.stderr}"
        return output.strip() or "(no output)"

    return Tool(
        name="shell_exec",
        description=(
            "Execute a shell command in a subprocess and return stdout/stderr. "
            "Process-isolated with a timeout, but NOT a security sandbox, and runs through a "
            "real shell (pipes/redirects work). Only for trusted code generation."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "the shell command to run"}},
            "required": ["command"],
        },
        fn=run,
    )
