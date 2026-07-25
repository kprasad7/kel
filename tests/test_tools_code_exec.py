import subprocess

from kel.tools import python_exec_tool


def test_python_exec_tool_wires_code_and_returns_stdout():
    captured = []

    def fake_run(cmd, timeout):
        captured.append((cmd, timeout))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="4\n", stderr="")

    tool = python_exec_tool(timeout=3.0, run_subprocess=fake_run)
    result = tool({"code": "print(2 + 2)"})

    assert result == "4"
    assert captured[0][1] == 3.0
    assert "print(2 + 2)" in captured[0][0]


def test_python_exec_tool_includes_stderr_and_exit_code_on_failure():
    def fake_run(cmd, timeout):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="NameError: x is not defined")

    tool = python_exec_tool(run_subprocess=fake_run)
    result = tool({"code": "print(x)"})

    assert "[exit code 1]" in result
    assert "NameError" in result


def test_python_exec_tool_handles_timeout():
    def fake_run(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    tool = python_exec_tool(timeout=2.0, run_subprocess=fake_run)
    result = tool({"code": "while True: pass"})

    assert "timed out" in result


def test_python_exec_tool_reports_no_output():
    def fake_run(cmd, timeout):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    tool = python_exec_tool(run_subprocess=fake_run)
    result = tool({"code": "x = 1"})

    assert result == "(no output)"


def test_python_exec_tool_actually_runs_real_python_subprocess():
    # no injected fake here — proves the real path genuinely works end to end
    tool = python_exec_tool(timeout=10.0)
    result = tool({"code": "print(21 * 2)"})
    assert result == "42"


def test_python_exec_tool_real_subprocess_enforces_timeout():
    tool = python_exec_tool(timeout=1.0)
    result = tool({"code": "import time; time.sleep(5)"})
    assert "timed out" in result
