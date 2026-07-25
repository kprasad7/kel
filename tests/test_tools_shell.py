import subprocess

from kel.tools import shell_exec_tool


def test_shell_exec_tool_wires_command_and_returns_stdout():
    captured = []

    def fake_run(command, timeout):
        captured.append((command, timeout))
        return subprocess.CompletedProcess(command, returncode=0, stdout="hello\n", stderr="")

    tool = shell_exec_tool(timeout=3.0, run_subprocess=fake_run)
    result = tool({"command": "echo hello"})

    assert result == "hello"
    assert captured == [("echo hello", 3.0)]


def test_shell_exec_tool_includes_stderr_and_exit_code_on_failure():
    def fake_run(command, timeout):
        return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr="command not found")

    tool = shell_exec_tool(run_subprocess=fake_run)
    result = tool({"command": "not-a-real-command"})

    assert "[exit code 127]" in result
    assert "command not found" in result


def test_shell_exec_tool_handles_timeout():
    def fake_run(command, timeout):
        raise subprocess.TimeoutExpired(command, timeout)

    tool = shell_exec_tool(timeout=2.0, run_subprocess=fake_run)
    result = tool({"command": "sleep 100"})

    assert "timed out" in result


def test_shell_exec_tool_actually_runs_real_shell_command():
    tool = shell_exec_tool(timeout=10.0)
    result = tool({"command": "echo hello-from-shell"})
    assert "hello-from-shell" in result
