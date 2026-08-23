"""
docker tool — wraps the Docker CLI as a LangChain tool.

The agent passes a raw docker command string (without the leading "docker")
and this tool executes it safely, returning stdout or a meaningful error.
"""

import subprocess
import shlex
from langchain_core.tools import tool


# Commands that mutate container/image state.  Listed for prompt awareness.
_DESTRUCTIVE_VERBS = {
    "rm", "rmi", "kill", "stop", "pause", "unpause",
    "restart", "prune", "remove", "run", "start",
    "push", "pull", "build", "tag", "commit",
    "cp", "rename", "update", "exec",
}


def _is_destructive(args: list[str]) -> bool:
    """Return True if the first meaningful argument is a mutating verb."""
    for arg in args:
        if not arg.startswith("-"):
            return arg.lower() in _DESTRUCTIVE_VERBS
    return False


@tool
def docker(command: str) -> str:
    """
    Run a docker command and return its output.

    Pass only the arguments that follow 'docker', e.g.:
        "ps -a"
        "images"
        "inspect my-container"
        "stats --no-stream"
        "network ls"
        "volume ls"

    Returns stdout on success or a descriptive error string on failure.
    Do NOT include 'docker' itself in the command string.
    """
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return f"[docker] Failed to parse command: {exc}"

    # Safety: strip any accidental leading 'docker' the LLM might add
    if args and args[0].lower() == "docker":
        args = args[1:]

    if not args:
        return "[docker] Empty command — nothing to run."

    full_cmd = ["docker"] + args

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=60,  # docker pulls / builds can be slow
        )
    except FileNotFoundError:
        return (
            "[docker] 'docker' binary not found. "
            "Make sure Docker is installed and on your PATH."
        )
    except subprocess.TimeoutExpired:
        return "[docker] Command timed out after 60 seconds."

    if result.returncode == 0:
        output = result.stdout.strip()
        return output if output else "[docker] Command succeeded with no output."

    stderr = result.stderr.strip()
    return (
        f"[docker] Command failed (exit {result.returncode}).\n"
        f"stderr: {stderr}"
    )
