"""
run_agent_ci.py — validate the agent in CI using the real ReAct loop.

The LLM (running via Ollama) decides when to call kubectl or docker,
calls the tool, reads the output, and produces a final answer — exactly
the same flow as the interactive agent.

Requires a model capable of following the ReAct text format, e.g.:
  qwen2.5:3b, llama3.2, mistral:7b  (NOT tinyllama)

Usage:
    python scripts/run_agent_ci.py \\
        --questions scripts/questions.txt \\
        --output    reports/agent-results.json

Environment variables:
    OLLAMA_BASE_URL       default: http://localhost:11434
    OLLAMA_MODEL          default: qwen2.5:3b
    AGENT_VERBOSE         default: false
    MAX_ITERATIONS        default: 3
    QUESTION_TIMEOUT_SECS default: 120
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

from agent import build_agent  # noqa: E402

QUESTION_TIMEOUT = int(os.getenv("QUESTION_TIMEOUT_SECS", "120"))


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[dict]:
    """
    Parse questions.txt.
    Each non-comment line: CATEGORY | question
    Returns list of dicts: {category, question}
    """
    questions = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                continue
            category, _, question = line.partition("|")
            questions.append({
                "category": category.strip().lower(),
                "question": question.strip(),
            })
    return questions


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_questions(questions: list[dict]) -> list[dict]:
    """
    Build one shared AgentExecutor and ask every question through the
    full ReAct loop — LLM reasons, calls tools, reads output, answers.
    """
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    print(f"Model  : {model}")
    print(f"Timeout: {QUESTION_TIMEOUT}s per question")
    print(f"Running: {len(questions)} question(s)\n")

    executor = build_agent()

    results = []
    for i, q in enumerate(questions, 1):
        category = q["category"]
        question = q["question"]

        print(f"[{i}/{len(questions)}] [{category.upper()}] {question}")

        start = time.monotonic()
        status = "pass"
        answer = ""
        error = ""
        tool_calls: list[str] = []

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(executor.invoke, {"input": question})
                try:
                    result = future.result(timeout=QUESTION_TIMEOUT)
                except FuturesTimeout:
                    status = "timeout"
                    error = f"No response within {QUESTION_TIMEOUT}s"
                    print(f"    ⏱  TIMEOUT after {QUESTION_TIMEOUT}s")
                    result = None

            if result is not None:
                answer = result.get("output", "").strip()
                for action, _ in result.get("intermediate_steps", []):
                    tool_name = getattr(action, "tool", None)
                    if tool_name and not tool_name.startswith("_"):
                        tool_calls.append(tool_name)

                if not answer:
                    status = "warn"
                    answer = "(empty response)"

        except Exception as exc:        # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        elapsed = round(time.monotonic() - start, 2)
        icon = {"pass": "✅", "warn": "⚠️", "timeout": "⏱", "error": "❌"}.get(status, "?")
        tools_str = ", ".join(tool_calls) if tool_calls else "none"
        print(f"    {icon} status={status}  tools=[{tools_str}]  time={elapsed}s")
        if answer:
            preview = answer[:120].replace("\n", " ")
            print(f"    → {preview}{'…' if len(answer) > 120 else ''}")
        if error and status not in ("timeout",):
            print(f"    ✗ {error[:120]}")
        print()

        results.append({
            "index": i,
            "category": category,
            "question": question,
            "status": status,           # pass | warn | timeout | error
            "answer": answer,
            "tool_calls": tool_calls,
            "elapsed_secs": elapsed,
            "error": error,
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
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "total": len(results),
        "pass":    sum(1 for r in results if r["status"] == "pass"),
        "warn":    sum(1 for r in results if r["status"] == "warn"),
        "timeout": sum(1 for r in results if r["status"] == "timeout"),
        "error":   sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # Timeouts and warns are acceptable — only hard errors fail CI
    if payload["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
