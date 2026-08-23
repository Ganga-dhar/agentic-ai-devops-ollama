"""
agent.py — LangChain ReAct agent backed by a local Ollama LLM.

The agent answers Kubernetes and Docker questions by calling real
kubectl and docker CLI commands as tools.

Usage:
    python src/agent.py                          # interactive REPL
    python src/agent.py "how many pods are running in all namespaces?"
"""

import os
import sys
import pathlib

# Ensure src/ is on sys.path so sibling modules (kubectl_tool, docker_tool)
# resolve regardless of the working directory the script is launched from.
# This must happen before the local imports below, hence the noqa markers.
_SRC_DIR = str(pathlib.Path(__file__).parent.resolve())
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

from langchain_ollama import ChatOllama  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain.agents import AgentExecutor, create_react_agent  # noqa: E402

from kubectl_tool import kubectl  # noqa: E402
from docker_tool import docker  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert DevOps assistant that specialises in Kubernetes and Docker.
You have access to the following tools that run REAL commands on the local machine:

  • kubectl — execute any kubectl subcommand to inspect or manage a Kubernetes cluster
  • docker  — execute any docker subcommand to inspect or manage containers and images

Guidelines:
  1. Always use the tools to fetch live data before answering questions about the
     current state of the cluster or Docker environment.
  2. Prefer read-only commands (get, describe, logs, inspect, ps, images, stats)
     unless the user explicitly asks you to make a change.
  3. Before running any destructive command (delete, rm, kill, drain, prune, etc.),
     explain what you are about to do and why, then proceed only if the user has
     clearly requested that action.
  4. If a command fails, show the error and try an alternative approach or ask the
     user for clarification.
  5. Present command output in a clean, readable format — use markdown tables or
     code blocks where they improve readability.
  6. Always explain what each command does and what the output means.
"""

# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------


def build_agent() -> AgentExecutor:
    """Construct and return the LangChain ReAct AgentExecutor."""

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0,          # deterministic for operational tasks
        num_predict=2048,
    )

    tools = [kubectl, docker]

    # Pull the standard ReAct prompt from LangChain hub and prepend our
    # system instructions by replacing the default prefix.
    # create_react_agent requires {tools}, {tool_names}, {input}, and
    # {agent_scratchpad} as plain string template variables.
    # MessagesPlaceholder does NOT work here — the ReAct text loop writes
    # a formatted string into agent_scratchpad, not a list of messages.
    prompt = ChatPromptTemplate.from_template(
        SYSTEM_PROMPT
        + "\n\nTools available:\n{tools}"
        + "\n\nTool names: {tool_names}"
        + "\n\n{input}"
        + "\n\n{agent_scratchpad}"
    )

    agent = create_react_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=AGENT_VERBOSE,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=(
            "I had trouble parsing my reasoning. "
            "Let me try a different approach."
        ),
        return_intermediate_steps=True,
    )

    return executor


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🤖  Agentic DevOps Assistant  (Ollama + LangChain)        ║
║   Model : {model:<48}║
║   Tools : kubectl, docker                                    ║
║   Type  : 'exit' or 'quit' to leave                         ║
╚══════════════════════════════════════════════════════════════╝
""".format(model=OLLAMA_MODEL)


def run_repl(executor: AgentExecutor) -> None:
    """Start an interactive question-answering loop."""
    print(BANNER)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            result = executor.invoke({"input": question})
            answer = result.get("output", "")
            print(f"\nAssistant: {answer}\n")
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    executor = build_agent()

    if len(sys.argv) > 1:
        # Single-shot mode: question passed as a CLI argument
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        result = executor.invoke({"input": question})
        print(f"Answer: {result.get('output', '')}")
    else:
        # Interactive REPL
        run_repl(executor)


if __name__ == "__main__":
    main()
