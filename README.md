# Agentic AI DevOps Assistant — Ollama + LangChain

[![CI](https://github.com/YOUR_USERNAME/agentic-ai-devops-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentic-ai-devops-ollama/actions/workflows/ci.yml)
[![Test Reports](https://img.shields.io/badge/Test%20Reports-GitHub%20Pages-blue)](https://YOUR_USERNAME.github.io/agentic-ai-devops-ollama/)

> Replace `YOUR_USERNAME` with your GitHub username in the two badge URLs above and in the Pages setup section below.

A conversational AI agent that answers Kubernetes and Docker questions by
calling **real** `kubectl` and `docker` commands as tools. The LLM runs
entirely locally via [Ollama](https://ollama.com) — no API keys required.

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

The agent uses LangChain's **ReAct** pattern — it reasons about which tool to
call, calls it, reads the output, and continues until it can produce a final
answer.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` |
| [Ollama](https://ollama.com) running locally | `ollama serve` |
| A pulled model | `ollama pull llama3.2` |
| `kubectl` on PATH | Only needed for K8s questions |
| `docker` on PATH | Only needed for Docker questions |

## Quick Start

```bash
# 1. Clone / enter the project
cd agentic-ai-devops-ollama

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# Edit .env if needed (model, Ollama URL, etc.)

# 5. Make sure Ollama is running and the model is available
ollama serve          # in a separate terminal if not already running
ollama pull llama3.2  # first time only

# 6. Run the agent
python src/agent.py
```

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

## Configuration

All settings are controlled via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled) |
| `AGENT_VERBOSE` | `true` | Show ReAct reasoning trace |
| `MAX_ITERATIONS` | `10` | Max tool-call iterations |

## Example Questions

**Kubernetes**
- "What pods are in a CrashLoopBackOff state?"
- "Show me the resource requests and limits for all pods in default namespace"
- "What events have happened in the kube-system namespace recently?"
- "Describe the service accounts in the default namespace"
- "What is the status of all nodes?"

**Docker**
- "List all running containers with their CPU and memory usage"
- "Which docker images are larger than 500MB?"
- "Show me the logs from the last 5 minutes of the nginx container"
- "What docker networks exist and what containers are connected to them?"
- "List all stopped containers"

## Project Structure

```
agentic-ai-devops-ollama/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main entry point — LLM + ReAct agent + REPL
│   ├── kubectl_tool.py   # kubectl CLI wrapper (@tool)
│   └── docker_tool.py    # docker CLI wrapper (@tool)
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
└── README.md
```

## Choosing a Model

Any model pulled in Ollama works. Models with stronger instruction-following
give better tool-use behaviour:

| Model | Notes |
|---|---|
| `llama3.2` | Good default, fast |
| `llama3.1:8b` | Slightly larger, better reasoning |
| `mistral` | Fast, good for structured output |
| `codellama` | Good for YAML / manifest generation |
| `qwen2.5:7b` | Strong tool-use capabilities |

```bash
ollama pull mistral
# then set OLLAMA_MODEL=mistral in .env
```

## Safety Notes

- **Read-only by default.** The system prompt instructs the LLM to prefer
  read-only commands and to explain any destructive action before running it.
- **No guardrails on the CLI level.** The tools will execute whatever command
  the LLM generates. Review the ReAct trace (`AGENT_VERBOSE=true`) to see
  exactly what commands are being run.
- **Do not run against production clusters** unless you fully understand and
  trust the model's output.

---

## CI / CD Pipeline

### What runs on every push / pull request

```
push / PR
    │
    ├─► lint          flake8 over src/ and tests/
    │
    ├─► unit-tests    pytest -m "not integration"   (no Docker/Kubernetes/Ollama needed)
    │       └─ uploads reports/unit-tests.html as artifact
    │
    ├─► integration-tests   pytest -m "integration"  (real Docker daemon on runner)
    │       └─ uploads reports/integration-tests.html as artifact
    │
    └─► build-report  merges both HTML reports → uploads GitHub Pages artifact
            │
            └─► pages.yml  deploys to GitHub Pages (main/master only)
```

### GitHub Actions workflows

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Lint → unit tests → integration tests → build Pages artifact |
| `.github/workflows/pages.yml` | Deploy the Pages artifact after CI succeeds on main/master |

### Enabling GitHub Pages

1. Push this repo to GitHub.
2. Go to **Settings → Pages**.
3. Under **Source**, choose **GitHub Actions**.
4. The next successful CI run on `main`/`master` will publish the report to:
   `https://YOUR_USERNAME.github.io/agentic-ai-devops-ollama/`

### Test structure

| Test file | Marker | What it tests |
|---|---|---|
| `tests/test_kubectl_tool.py` | _(unit)_ | Arg parsing, prefix stripping, error paths, `_is_destructive` |
| `tests/test_docker_tool.py` | _(unit)_ + `integration` | Same as kubectl + real `docker ps/images/network ls` |
| `tests/test_agent_config.py` | _(unit)_ | Env var defaults, type coercions, `build_agent()` construction |

### Running tests locally

```bash
# Unit tests only (no Docker / Ollama needed)
pytest -m "not integration"

# Unit + integration (requires Docker daemon)
pytest

# With HTML report
pytest --html=reports/test-report.html --self-contained-html
```
