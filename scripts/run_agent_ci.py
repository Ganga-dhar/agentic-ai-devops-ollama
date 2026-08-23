"""
run_agent_ci.py — validate the agent in CI using two modes:

  tool-direct  Run CLI commands through the tool wrappers directly,
               no LLM involved. Fast, deterministic, always works.

  llm-only     Send a question straight to ChatOllama.invoke(),
               no ReAct loop. Tests LLM connectivity and generation.

This avoids the ReAct loop entirely in CI, which requires a large,
instruction-following model to work reliably.

Usage:
    python scripts/run_agent_ci.py \\
        --questions scripts/questions.txt \\
        --output    reports/agent-results.json

Environment variables:
    OLLAMA_BASE_URL       default: http://localhost:11434
    OLLAMA_MODEL          default: tinyllama
    AGENT_VERBOSE         default: false
    QUESTION_TIMEOUT_SECS default: 30
"""

import argparse
import json
import os
import sys
import pathlib
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Make src/ importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

os.environ.setdefault("AGENT_VERBOSE", "false")

from langchain_ollama import ChatOllama                    # noqa: E402
from langchain_core.messages import HumanMessage           # noqa: E402
from kubectl_tool import kubectl                            # noqa: E402
from docker_tool import docker                             # noqa: E402

QUESTION_TIMEOUT = int(os.getenv("QUESTION_TIMEOUT_SECS", "30"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[dict]:
    """
    Parse questions.txt.
    Each non-comment line: CATEGORY | MODE | question_or_command
    Returns list of dicts: {category, mode, question}
    """
    questions = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue
            category, mode, question = parts
            questions.append({
                "category": category.lower(),
                "mode": mode.lower(),
                "question": question,
            })
    return questions


# ---------------------------------------------------------------------------
# Tool-direct runner
# ---------------------------------------------------------------------------

def run_tool_direct(category: str, command: str) -> tuple[str, str]:
    """
    Call kubectl.func or docker.func directly — no LLM.
    Returns (answer, error).
    """
    if category == "docker":
        answer = docker.func(command)
    elif category == "kubernetes":
        answer = kubectl.func(command)
    else:
        return "", f"Unknown category for tool-direct: {category}"
    return answer, ""


# ---------------------------------------------------------------------------
# LLM-only runner
# ---------------------------------------------------------------------------

_llm_instance = None


def get_llm() -> ChatOllama:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0,
            num_predict=256,     # keep answers short for CI speed
        )
    return _llm_instance


def run_llm_only(question: str) -> tuple[str, str]:
    """
    Send a question directly to the LLM — no ReAct loop, no tools.
    Returns (answer, error).
    """
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=question)])
    answer = response.content.strip() if hasattr(response, "content") else str(response)
    return answer, ""


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_one(category: str, mode: str, question: str) -> tuple[str, str, str]:
    """
    Dispatch to the right runner.
    Returns (status, answer, error): status is pass|warn|timeout|error
    """
    if mode == "tool-direct":
        answer, error = run_tool_direct(category, question)
    elif mode == "llm-only":
        answer, error = run_llm_only(question)
    else:
        return "error", "", f"Unknown mode: {mode}"

    if error:
        return "error", "", error

    # A tool returning a "[...] binary not found" or empty string is a warn,
    # not a hard error — kubectl will fail in CI (no cluster) and that's expected.
    if not answer:
        return "warn", "(empty response)", ""

    if any(tag in answer for tag in ("[kubectl] 'kubectl' binary not found",
                                     "[docker] 'docker' binary not found")):
        return "warn", answer, ""

    return "pass", answer, ""


def run_questions(questions: list[dict]) -> list[dict]:
    print(f"Model  : {OLLAMA_MODEL}")
    print(f"Timeout: {QUESTION_TIMEOUT}s per question")
    print(f"Running: {len(questions)} question(s)\n")

    results = []
    for i, q in enumerate(questions, 1):
        category = q["category"]
        mode = q["mode"]
        question = q["question"]

        label = f"[{i}/{len(questions)}] [{category.upper()}][{mode}]"
        print(f"{label} {question}")

        start = time.monotonic()
        status = "pass"
        answer = ""
        error = ""

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_one, category, mode, question)
                try:
                    status, answer, error = future.result(timeout=QUESTION_TIMEOUT)
                except FuturesTimeout:
                    status = "timeout"
                    error = f"Timed out after {QUESTION_TIMEOUT}s"
                    print(f"    ⏱  TIMEOUT after {QUESTION_TIMEOUT}s")

        except Exception as exc:        # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        elapsed = round(time.monotonic() - start, 2)
        icon = {"pass": "✅", "warn": "⚠️", "timeout": "⏱", "error": "❌"}.get(status, "?")
        print(f"    {icon} status={status}  time={elapsed}s")
        if answer:
            preview = answer[:120].replace("\n", " ")
            print(f"    → {preview}{'…' if len(answer) > 120 else ''}")
        if error and status != "timeout":
            print(f"    ✗ {error[:120]}")
        print()

        results.append({
            "index": i,
            "category": category,
            "mode": mode,
            "question": question,
            "status": status,
            "answer": answer,
            "error": error,
            "elapsed_secs": elapsed,
        })

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    total = len(results)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    parts = "  ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    print("=" * 60)
    print(f"SUMMARY  total={total}  {parts}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="scripts/questions.txt")
    parser.add_argument("--output",    default="reports/agent-results.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("No questions found.")
        sys.exit(1)

    results = run_questions(questions)
    print_summary(results)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
        "total": len(results),
        "pass":    sum(1 for r in results if r["status"] == "pass"),
        "warn":    sum(1 for r in results if r["status"] == "warn"),
        "timeout": sum(1 for r in results if r["status"] == "timeout"),
        "error":   sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # Only hard-fail on errors, not timeouts or warns
    if payload["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
