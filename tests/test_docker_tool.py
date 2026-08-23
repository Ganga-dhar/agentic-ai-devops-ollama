"""
Unit + integration tests for src/docker_tool.py

Unit tests  — mock subprocess.run, no Docker daemon needed.
Integration — marked with @pytest.mark.integration; run against the real
              Docker daemon present on GitHub-hosted runners.
              Skipped automatically when Docker is unavailable.
"""

import subprocess
import sys
import pathlib
import shutil
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from docker_tool import docker, _is_destructive  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(command: str) -> str:
    return docker.func(command)


def _make_completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


DOCKER_AVAILABLE = shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_malformed_quotes_returns_parse_error(self):
        result = _invoke("ps --filter 'name=bad")
        assert result.startswith("[docker] Failed to parse command:")

    def test_strips_leading_docker_prefix(self):
        with patch("subprocess.run", return_value=_make_completed(0, "stripped")) as mock_run:
            result = _invoke("docker ps")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd.count("docker") == 1
        assert result == "stripped"

    def test_strips_leading_docker_prefix_case_insensitive(self):
        with patch("subprocess.run", return_value=_make_completed(0, "ok")) as mock_run:
            _invoke("Docker images")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["docker", "images"]

    def test_empty_command_returns_guard_message(self):
        result = _invoke("")
        assert result == "[docker] Empty command — nothing to run."

    def test_whitespace_only_returns_guard_message(self):
        result = _invoke("   ")
        assert result == "[docker] Empty command — nothing to run."


# ---------------------------------------------------------------------------
# subprocess error paths
# ---------------------------------------------------------------------------

class TestSubprocessErrors:
    def test_binary_not_found_returns_helpful_message(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _invoke("ps")
        assert "[docker] 'docker' binary not found" in result
        assert "PATH" in result

    def test_timeout_returns_timeout_message(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=60),
        ):
            result = _invoke("pull ubuntu:latest")
        assert "[docker] Command timed out after 60 seconds." == result

    def test_nonzero_exit_returns_failure_message_with_stderr(self):
        mock_cp = _make_completed(returncode=1, stderr="No such container: xyz")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("inspect xyz")
        assert "[docker] Command failed (exit 1)" in result
        assert "No such container: xyz" in result

    def test_nonzero_exit_includes_exit_code(self):
        mock_cp = _make_completed(returncode=125, stderr="daemon not running")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("ps")
        assert "exit 125" in result


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestSuccessfulExecution:
    def test_returns_stripped_stdout(self):
        output = "CONTAINER ID   IMAGE   STATUS\nabc123   nginx   Up 2 hours\n"
        mock_cp = _make_completed(0, output)
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("ps")
        assert result == output.strip()

    def test_empty_stdout_returns_no_output_message(self):
        mock_cp = _make_completed(0, "")
        with patch("subprocess.run", return_value=mock_cp):
            result = _invoke("network prune -f")
        assert result == "[docker] Command succeeded with no output."

    def test_correct_command_is_assembled(self):
        mock_cp = _make_completed(0, "REPOSITORY   TAG")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("images --format table")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["docker", "images", "--format", "table"]

    def test_subprocess_called_with_capture_and_timeout(self):
        mock_cp = _make_completed(0, "ok")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("ps -a")
        _, kwargs = mock_run.call_args
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") == 60   # docker uses 60s, not 30s

    def test_filter_flag_preserved(self):
        mock_cp = _make_completed(0, "abc123")
        with patch("subprocess.run", return_value=mock_cp) as mock_run:
            _invoke("ps --filter status=running -q")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["docker", "ps", "--filter", "status=running", "-q"]


# ---------------------------------------------------------------------------
# Integration tests (real Docker daemon — GitHub Actions ubuntu runner)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available on this machine")
class TestDockerIntegration:
    """
    These tests call the real docker CLI against the daemon on the runner.
    They validate that the tool wraps real output correctly without errors.
    All commands are strictly read-only.
    """

    def test_docker_version_succeeds(self):
        result = _invoke("version --format '{{.Client.Version}}'")
        assert not result.startswith("[docker] Command failed")
        assert not result.startswith("[docker] 'docker' binary not found")

    def test_docker_ps_returns_string(self):
        result = _invoke("ps -a")
        assert isinstance(result, str)
        assert not result.startswith("[docker] Command failed")
        # Header line always present even with no containers
        assert "CONTAINER" in result or result == "[docker] Command succeeded with no output."

    def test_docker_images_returns_string(self):
        result = _invoke("images")
        assert isinstance(result, str)
        assert not result.startswith("[docker] Command failed")

    def test_docker_info_succeeds(self):
        result = _invoke("info --format '{{.ServerVersion}}'")
        assert isinstance(result, str)
        assert not result.startswith("[docker] Command failed")

    def test_docker_network_ls_returns_string(self):
        result = _invoke("network ls")
        assert isinstance(result, str)
        assert not result.startswith("[docker] Command failed")
        assert "NETWORK ID" in result

    def test_docker_volume_ls_returns_string(self):
        result = _invoke("volume ls")
        assert isinstance(result, str)
        assert not result.startswith("[docker] Command failed")


# ---------------------------------------------------------------------------
# _is_destructive helper
# ---------------------------------------------------------------------------

DESTRUCTIVE_CASES = [
    (["rm", "my-container"], True),
    (["rmi", "my-image:latest"], True),
    (["kill", "abc123"], True),
    (["stop", "web"], True),
    (["prune", "--all"], True),
    (["run", "-d", "nginx"], True),
    (["build", "-t", "myapp", "."], True),
]

READ_ONLY_CASES = [
    (["ps", "-a"], False),
    (["images"], False),
    (["inspect", "my-container"], False),
    (["logs", "web", "--tail=50"], False),
    (["network", "ls"], False),
    (["stats", "--no-stream"], False),
]


@pytest.mark.parametrize("args,expected", DESTRUCTIVE_CASES)
def test_is_destructive_returns_true(args, expected):
    assert _is_destructive(args) is expected


@pytest.mark.parametrize("args,expected", READ_ONLY_CASES)
def test_is_destructive_returns_false_for_readonly(args, expected):
    assert _is_destructive(args) is expected


def test_is_destructive_empty_args():
    assert _is_destructive([]) is False
