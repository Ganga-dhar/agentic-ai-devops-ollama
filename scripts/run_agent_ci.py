"""
run_agent_ci.py — fire pre-written questions at the agent in CI and
save structured JSON results.

Usage:
    python scripts/run_agent_ci.py \
        --questions scripts/questions.txt \
        --output    reports/agent-results.json

Environment variables (same as the agent):
    OLLAMA_BASE_URL       default: http://localhost:11434
    OLLAMA_MODEL          default: llama3.2
    AGENT_VERBOSE         default: false  (suppressed in CI for clean output)
    MAX_ITERATIONS        default: 10
    QUESTION_TIMEOUT_SECS default: 90  (per-question hard timeout)
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

os.environ.setdefault("AGENT_VERBOSE", "false")   # quieter CI output

from agent import build_agent  # noqa: E402

QUESTION_TIMEOUT = int(os.getenv("QUESTION_TIMEOUT_SECS", "90"))


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[dict]:
    """
    Parse questions.txt.  Each non-comment line is:
        CATEGORY | question text
    Returns a list of dicts: {category, question}
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
    Build one shared AgentExecutor and ask every question.
    Returns a list of result dicts.
    """
    print(f"Building agent (model: {os.getenv('OLLAMA_MODEL', 'llama3.2')}) …")
    executor = build_agent()
    print(f"Agent ready. Running {len(questions)} question(s).\n")

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
            # Run invoke in a thread so we can enforce a hard wall-clock timeout.
            # executor.invoke() has no built-in timeout and will hang indefinitely
            # if Ollama is slow or unresponsive on the CI runner.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(executor.invoke, {"input": question})
                try:
                    result = future.result(timeout=QUESTION_TIMEOUT)
                except FuturesTimeout:
                    raise TimeoutError(
                        f"Question timed out after {QUESTION_TIMEOUT}s"
                    )

            answer = result.get("output", "").strip()

            # Extract tool names from intermediate steps
            for action, _ in result.get("intermediate_steps", []):
                tool_calls.append(getattr(action, "tool", str(action)))

            if not answer:
                status = "warn"
                answer = "(empty response)"

        except TimeoutError as exc:
            status = "timeout"
            error = str(exc)
            answer = ""
            print(f"    ⏱ TIMEOUT after {QUESTION_TIMEOUT}s")

        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            answer = ""
            traceback.print_exc()

        elapsed = round(time.monotonic() - start, 2)
        print(f"    → status={status}  tools={tool_calls}  time={elapsed}s")
        if answer:
            # Print first 120 chars as a preview
            preview = answer[:120].replace("\n", " ")
            print(f"    → {preview}{'…' if len(answer) > 120 else ''}")
        print()

        results.append({
            "index": i,
            "category": category,
            "question": question,
            "status": status,       # pass | warn | timeout | error
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
    parser = argparse.ArgumentParser(description="Run agent validation questions in CI")
    parser.add_argument(
        "--questions",
        default="scripts/questions.txt",
        help="Path to questions file",
    )
    parser.add_argument(
        "--output",
        default="reports/agent-results.json",
        help="Path to write JSON results",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("No questions found — check the questions file.")
        sys.exit(1)

    results = run_questions(questions)
    print_summary(results)

    # Write results JSON
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "total": len(results),
        "pass": sum(1 for r in results if r["status"] == "pass"),
        "warn": sum(1 for r in results if r["status"] == "warn"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # Exit non-zero if any question errored or timed out
    bad_count = sum(1 for r in results if r["status"] in {"error", "timeout"})
    if bad_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
