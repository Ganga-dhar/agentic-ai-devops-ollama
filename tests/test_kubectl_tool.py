"""
Unit tests for src/kubectl_tool.py

Covers:
  - shlex parsing errors
  - stripping accidental 'kubectl' prefix
  - empty command guard
  - kubectl binary not found (FileNotFoundError)
  - command timeout (TimeoutExpired)
  - non-zero exit code with stderr
  - successful command with stdout
  - successful command with empty stdout
  - _is_destructive() helper (parametrized)
"""

import subprocess
import sys
import pathlib
import pytest
from unittest.mock import patch, MagicMock

# Make src/ importable regardless of working directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from kubectl_tool import kubectl, _is_destructive  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(command: str) -> str:
    """Call the underlying function, bypassing the @tool wrapper."""
    return kubectl.func(command)


def _make_completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_malformed_quotes_returns_parse_error(self):
        result = _invoke("get pods -n 'bad quote")
        assert result.startswith("[kubectl] Failed to parse command:")

    def test_strips_leading_kubectl_prefix(self):
        """LLM sometimes includes 'kubectl' itself — it should be stripped."""
        with patch("subprocess.run", return_value=_make_completed(0, "stripped")) as mock_run:
            result = _invoke("kubectl get pods")
        called_cmd = mock_run.call_args[0][0]
        # Must NOT have ['kubectl', 'kubectl', ...]
        assert called_cmd.count("kubectl") == 1
        assert result == "stripped"

    def test_strips_leading_kubectl_prefix_case_insensitive(self):
        with patch("subprocess.run", return_value=_make_completed(0, "ok")) as mock_run:
            _invoke("Kubectl get nodes")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["kubectl", "get", "nodes"]

    def test_empty_command_returns_guard_message(self):
        result = _invoke("")
        assert result == "[kubectl] Empty command — nothing to run."

    def test_whitespace_only_command_returns_guard_message(self):
        result = _invoke("   ")
        assert result == "[kubectl] Empty command — nothing to run."


# ---------------------------------------------------------------------------
# subprocess error paths
# ---------------------------------------------------------------------------

class TestSubprocessErrors:
    def test_binary_not_found_returns_helpful_message(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _invoke("get pods")
        assert "[kubectl] 'kubectl' binary not found" in result
        assert "PATH" in result

    def test_timeout_returns_timeout_message(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="kubectl", timeout=30),
        ):
            result = _invoke("get pods")
        assert "[kubectl] Command timed out after 30 seconds." == result

    def test_nonzero_exit_returns_failure_message_with_stderr(self):
        mock_cp = _make_completed(returncode=1, stderr="Error from server: Forbidden")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("get secrets")
        assert "[kubectl] Command failed (exit 1)" in result
        assert "Error from server: Forbidden" in result

    def test_nonzero_exit_includes_exit_code(self):
        mock_cp = _make_completed(returncode=128, stderr="fatal error")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("version")
        assert "exit 128" in result


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestSuccessfulExecution:
    def test_returns_stripped_stdout(self):
        mock_cp = _make_completed(0, "  NAME   READY   STATUS\nnginx   1/1   Running\n")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("get pods")
        assert result == "NAME   READY   STATUS\nnginx   1/1   Running"

    def test_empty_stdout_returns_no_output_message(self):
        mock_cp = _make_completed(0, "")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("apply -f deployment.yaml")
        assert result == "[kubectl] Command succeeded with no output."

    def test_correct_command_is_assembled(self):
        mock_cp = _make_completed(0, "v1.29.0")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("version --short")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["kubectl", "version", "--short"]

    def test_subprocess_called_with_capture_and_timeout(self):
        mock_cp = _make_completed(0, "ok")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("get nodes")
        _, kwargs = mock_run.call_args
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") == 30

    def test_namespace_flag_preserved(self):
        mock_cp = _make_completed(0, "pod-123")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("get pods -n kube-system -o wide")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["kubectl", "get", "pods", "-n", "kube-system", "-o", "wide"]


# ---------------------------------------------------------------------------
# _is_destructive helper
# ---------------------------------------------------------------------------

DESTRUCTIVE_CASES = [
    (["delete", "pod", "nginx"], True),
    (["drain", "node-1", "--ignore-daemonsets"], True),
    (["apply", "-f", "deploy.yaml"], True),
    (["scale", "deployment/web", "--replicas=0"], True),
    (["create", "configmap", "my-cfg"], True),
]

READ_ONLY_CASES = [
    (["get", "pods"], False),
    (["describe", "node", "worker-1"], False),
    (["logs", "nginx-pod", "--tail=100"], False),
    (["version"], False),
    (["cluster-info"], False),
]

FLAG_FIRST_CASES = [
    (["-n", "kube-system", "get", "pods"], False),
    (["--namespace=default", "get", "nodes"], False),
]


@pytest.mark.parametrize("args,expected", DESTRUCTIVE_CASES)
def test_is_destructive_returns_true(args, expected):
    assert _is_destructive(args) is expected


@pytest.mark.parametrize("args,expected", READ_ONLY_CASES)
def test_is_destructive_returns_false_for_readonly(args, expected):
    assert _is_destructive(args) is expected


@pytest.mark.parametrize("args,expected", FLAG_FIRST_CASES)
def test_is_destructive_skips_flags(args, expected):
    assert _is_destructive(args) is expected


def test_is_destructive_empty_args():
    assert _is_destructive([]) is False
