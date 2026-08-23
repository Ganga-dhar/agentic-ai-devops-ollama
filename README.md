# Agentic AI DevOps Assistant — Ollama + LangChain

[![CI](https://github.com/YOUR_USERNAME/agentic-ai-devops-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentic-ai-devops-ollama/actions/workflows/ci.yml)
[![Test Reports](https://img.shields.io/badge/Test%20Reports-GitHub%20Pages-blue)](https://YOUR_USERNAME.github.io/agentic-ai-devops-ollama/)

> Replace `YOUR_USERNAME` with your GitHub username in the two badge URLs above.

A conversational AI agent that answers Kubernetes and Docker questions by calling **real** `kubectl` and `docker` commands as tools. The LLM runs entirely locally via [Ollama](https://ollama.com) — no API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     agent.py                        │
│                                                     │
│  User question                                      │
│       │                                             │
│       ▼                                             │
│  ChatOllama (llama3.2 / any Ollama model)           │
│       │  ReAct loop                                 │
│       ▼                                             │
│  ┌─────────────┐        ┌─────────────────────┐    │
│  │ kubectl tool│        │    docker tool      │    │
│  │  (kubectl …)│        │    (docker …)       │    │
│  └──────┬──────┘        └──────────┬──────────┘    │
│         │                          │                │
│         ▼                          ▼                │
│      kubectl CLI              docker CLI            │
│   (real cluster calls)    (real daemon calls)       │
└─────────────────────────────────────────────────────┘
```

The agent uses LangChain's **ReAct** pattern — it reasons about which tool to call, calls it, reads the output, and continues until it can produce a final answer.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` |
| [Ollama](https://ollama.com) running locally | `ollama serve` |
| A pulled model | `ollama pull llama3.2` |
| `kubectl` on PATH | Only needed for Kubernetes questions |
| `docker` on PATH | Only needed for Docker questions |

---

## Quick Start

```bash
# 1. Clone / enter the project
cd agentic-ai-devops-ollama

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
# Edit .env to set your model and Ollama URL

# 5. Start Ollama and pull a model
ollama serve                    # separate terminal if not already running
ollama pull llama3.2            # first time only

# 6. Run the agent
python src/agent.py
```

---

## Usage Modes

### Interactive REPL

```bash
python src/agent.py
```

```
You: how many pods are running across all namespaces?
You: show me all docker containers that are stopped
You: what nodes are in my cluster and what is their status?
You: describe the kube-system namespace
```

### Single-shot (CLI argument)

```bash
python src/agent.py "list all deployments in the default namespace"
python src/agent.py "show running docker containers with their port mappings"
```

---

## Configuration

All settings are controlled via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled first) |
| `AGENT_VERBOSE` | `true` | Show ReAct reasoning trace |
| `MAX_ITERATIONS` | `10` | Max tool-call iterations per question |

---

## Example Questions

**Kubernetes**
- "What pods are in a CrashLoopBackOff state?"
- "Show me the resource requests and limits for all pods in default namespace"
- "What events have happened in the kube-system namespace recently?"
- "What is the status of all nodes?"

**Docker**
- "List all running containers with their CPU and memory usage"
- "Which docker images are larger than 500 MB?"
- "What docker networks exist and what containers are connected to them?"
- "List all stopped containers"

---

## Choosing a Model

Any model pulled in Ollama works. Models with stronger instruction-following give better tool-use behaviour:

| Model | Notes |
|---|---|
| `llama3.2` | Good default, fast |
| `llama3.1:8b` | Better reasoning, slightly larger |
| `mistral` | Fast, good for structured output |
| `codellama` | Good for YAML / manifest generation |
| `qwen2.5:7b` | Strong tool-use capabilities |

```bash
ollama pull mistral
# then set OLLAMA_MODEL=mistral in .env
```

---

## Safety Notes

- **Read-only by default.** The system prompt instructs the LLM to prefer read-only commands and explain any destructive action before running it.
- **No CLI-level guardrails.** The tools execute whatever command the LLM generates. Keep `AGENT_VERBOSE=true` to see every command being run.
- **Do not run against production clusters** unless you fully understand and trust the model's output.

---

## Project Structure

```
agentic-ai-devops-ollama/
├── .github/
│   └── workflows/
│       └── ci.yml              # Full CI + GitHub Pages pipeline
├── scripts/
│   ├── questions.txt           # Validation questions for CI
│   ├── run_agent_ci.py         # CI runner (tool-direct + llm-only modes)
│   └── build_agent_report.py  # Builds interactive HTML report from JSON
├── src/
│   ├── __init__.py
│   ├── agent.py                # Entry point — LLM + ReAct agent + REPL
│   ├── kubectl_tool.py         # kubectl CLI wrapper (@tool)
│   └── docker_tool.py          # docker CLI wrapper (@tool)
├── tests/
│   ├── test_kubectl_tool.py    # Unit tests for kubectl tool
│   ├── test_docker_tool.py     # Unit + Docker integration tests
│   └── test_agent_config.py   # Agent config / env var tests
├── .env.example                # Environment variable template
├── .gitignore
├── pytest.ini
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev / CI dependencies
└── README.md
```

---

## CI / CD Pipeline

Every push and pull request to `main` runs a 6-job pipeline in GitHub Actions and publishes results to GitHub Pages.

### Pipeline flow

```
push / PR
    │
    ├─► lint                  flake8 over src/ and tests/
    │
    ├─► unit-tests            pytest -m "not integration"
    │       └─ HTML report artifact
    │
    ├─► integration-tests     pytest -m "integration"  (real Docker daemon)
    │       └─ HTML report artifact
    │
    ├─► agent-validation      Ollama runs inside CI, questions fired in two modes
    │       └─ HTML report artifact + JSON results
    │
    └─► build-report          merge all 3 reports + index.html → Pages artifact
            │
            └─► deploy        actions/deploy-pages → GitHub Pages
```

### Agent validation in CI

The `agent-validation` job installs Ollama on the runner, pulls `tinyllama`, then runs `scripts/run_agent_ci.py` against `scripts/questions.txt`.

Questions use two modes that avoid the ReAct loop (which requires a large model):

| Mode | What it does | Used for |
|---|---|---|
| `tool-direct` | Calls `docker.func()` / `kubectl.func()` directly — no LLM | Docker and Kubernetes commands |
| `llm-only` | Sends question straight to `ChatOllama.invoke()` — no tool loop | Reasoning / knowledge questions |

Example `questions.txt` entry:
```
docker     | tool-direct | ps -a
reasoning  | llm-only    | In one paragraph, what is the difference between a Docker container and a Kubernetes Pod?
```

Add or edit questions in `scripts/questions.txt` to extend the validation suite.

### GitHub Pages reports

After a successful CI run on `main`, three reports are published to:

```
https://YOUR_USERNAME.github.io/agentic-ai-devops-ollama/
```

| Report | Content |
|---|---|
| `unit-tests.html` | pytest results with coverage for all unit tests |
| `integration-tests.html` | pytest results for real Docker CLI integration tests |
| `agent-validation.html` | Interactive Q&A report — filter by status or category |

The agent validation report shows each question's answer, mode, elapsed time, and pass/warn/timeout/error status with filter buttons.

### Enabling GitHub Pages (one-time setup)

1. Push this repo to GitHub.
2. Go to **Settings → Pages**.
3. Under **Source**, choose **GitHub Actions**.
4. The next CI run on `main` publishes the reports automatically.

---

## Running Tests Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Unit tests only (no Docker / Ollama needed)
pytest -m "not integration"

# Unit + integration (requires Docker daemon)
pytest

# With HTML report
pytest --html=reports/test-report.html --self-contained-html

# Run agent validation locally (requires Ollama running)
python scripts/run_agent_ci.py \
    --questions scripts/questions.txt \
    --output    reports/agent-results.json

python scripts/build_agent_report.py \
    --input  reports/agent-results.json \
    --output reports/agent-validation.html
```

---

## GitHub Actions Workflow Reference

The single workflow file `.github/workflows/ci.yml` contains all 6 jobs:

| Job | Trigger | What it does |
|---|---|---|
| `lint` | every push/PR | flake8 — max line length 100, ignores E501/W503 |
| `unit-tests` | after lint | pytest unit tests + coverage XML + HTML report |
| `integration-tests` | after lint | pytest integration tests against real Docker daemon |
| `agent-validation` | after lint | installs Ollama, pulls tinyllama, runs questions, builds HTML report |
| `build-report` | after all 3 test jobs | downloads artifacts, generates index.html, uploads Pages artifact |
| `deploy` | after build-report (main only) | deploys to GitHub Pages via `actions/deploy-pages` |
