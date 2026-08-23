"""
kubectl tool — wraps the kubectl CLI as a LangChain tool.

The agent passes a raw kubectl command string (without the leading "kubectl")
and this tool executes it safely, returning stdout or a meaningful error.
"""

import subprocess
import shlex
from langchain_core.tools import tool


# Commands that mutate cluster state.  The agent may still call them, but
# they are listed here so the prompt can warn the LLM to be careful.
_DESTRUCTIVE_VERBS = {
    "delete", "drain", "cordon", "uncordon",
    "patch", "replace", "apply", "create", "edit",
    "scale", "rollout", "label", "annotate", "taint",
}


def _is_destructive(args: list[str]) -> bool:
    """Return True if the first meaningful argument is a mutating verb."""
    for arg in args:
        if not arg.startswith("-"):
            return arg.lower() in _DESTRUCTIVE_VERBS
    return False


@tool
def kubectl(command: str) -> str:
    """
    Run a kubectl command and return its output.

    Pass only the arguments that follow 'kubectl', e.g.:
        "get pods -n default"
        "describe node my-node"
        "logs deployment/my-app --tail=50"

    Returns stdout on success or a descriptive error string on failure.
    Do NOT include 'kubectl' itself in the command string.
    """
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return f"[kubectl] Failed to parse command: {exc}"

    # Safety: strip any accidental leading 'kubectl' the LLM might add
    if args and args[0].lower() == "kubectl":
        args = args[1:]

    if not args:
        return "[kubectl] Empty command — nothing to run."

    full_cmd = ["kubectl"] + args

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return (
            "[kubectl] 'kubectl' binary not found. "
            "Make sure kubectl is installed and on your PATH."
        )
    except subprocess.TimeoutExpired:
        return "[kubectl] Command timed out after 30 seconds."

    if result.returncode == 0:
        output = result.stdout.strip()
        return output if output else "[kubectl] Command succeeded with no output."

    stderr = result.stderr.strip()
    return (
        f"[kubectl] Command failed (exit {result.returncode}).\n"
        f"stderr: {stderr}"
    )
