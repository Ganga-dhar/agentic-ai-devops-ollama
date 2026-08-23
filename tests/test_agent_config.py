"""
Unit tests for configuration parsing in src/agent.py.

These tests verify that environment variables are read correctly,
defaults are applied, and type coercions work — without starting
Ollama or making any network calls.

The agent module imports ChatOllama at module level, so we patch
langchain_ollama.ChatOllama before importing agent to avoid
ConnectionRefusedError in CI.
"""

import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch
from langchain.agents import AgentExecutor

# Make src/ importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_agent_with_env(monkeypatch, env: dict) -> object:
    """
    Reload the agent module after patching environment variables.
    Returns the freshly imported module.
    """
    # Apply env overrides
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # Remove cached module so module-level code re-runs
    for mod_name in list(sys.modules.keys()):
        if "agent" in mod_name and "test" not in mod_name:
            del sys.modules[mod_name]

    with patch("langchain_ollama.ChatOllama", MagicMock()):
        import agent as ag
    return ag


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_ollama_base_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        ag = _reload_agent_with_env(monkeypatch, {})
        assert ag.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_default_ollama_model(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        ag = _reload_agent_with_env(monkeypatch, {})
        assert ag.OLLAMA_MODEL == "llama3.2"

    def test_default_agent_verbose_is_true(self, monkeypatch):
        monkeypatch.delenv("AGENT_VERBOSE", raising=False)
        ag = _reload_agent_with_env(monkeypatch, {})
        assert ag.AGENT_VERBOSE is True

    def test_default_max_iterations(self, monkeypatch):
        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        ag = _reload_agent_with_env(monkeypatch, {})
        assert ag.MAX_ITERATIONS == 10


# ---------------------------------------------------------------------------
# Custom values
# ---------------------------------------------------------------------------

class TestCustomValues:
    def test_custom_ollama_base_url(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {"OLLAMA_BASE_URL": "http://192.168.1.10:11434"})
        assert ag.OLLAMA_BASE_URL == "http://192.168.1.10:11434"

    def test_custom_ollama_model(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {"OLLAMA_MODEL": "mistral"})
        assert ag.OLLAMA_MODEL == "mistral"

    def test_custom_max_iterations(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {"MAX_ITERATIONS": "25"})
        assert ag.MAX_ITERATIONS == 25

    def test_max_iterations_is_int(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {"MAX_ITERATIONS": "5"})
        assert isinstance(ag.MAX_ITERATIONS, int)


# ---------------------------------------------------------------------------
# AGENT_VERBOSE boolean parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("true",  True),
    ("True",  True),
    ("TRUE",  True),
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("0",     False),
    ("1",     False),   # only "true" (case-insensitive) is truthy
])
def test_agent_verbose_parsing(monkeypatch, value, expected):
    ag = _reload_agent_with_env(monkeypatch, {"AGENT_VERBOSE": value})
    assert ag.AGENT_VERBOSE is expected


# ---------------------------------------------------------------------------
# build_agent — construction, no network call
# ---------------------------------------------------------------------------

class TestBuildAgent:
    def test_build_agent_returns_executor(self, monkeypatch):
        """build_agent() should return an AgentExecutor without contacting Ollama."""
        from langchain.agents import AgentExecutor

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("langchain_ollama.ChatOllama", return_value=mock_llm):
            ag = _reload_agent_with_env(monkeypatch, {})
            with patch("langchain_ollama.ChatOllama", return_value=mock_llm):
                with patch("langchain.agents.create_react_agent") as mock_create:
                    mock_create.return_value = MagicMock()
                    executor = ag.build_agent()

        assert isinstance(executor, AgentExecutor)

    def test_build_agent_registers_two_tools(self, monkeypatch):
        """The executor must have exactly kubectl and docker registered."""
        from langchain.agents import AgentExecutor

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("langchain_ollama.ChatOllama", return_value=mock_llm):
            ag = _reload_agent_with_env(monkeypatch, {})
            with patch("langchain_ollama.ChatOllama", return_value=mock_llm):
                with patch("langchain.agents.create_react_agent") as mock_create:
                    mock_create.return_value = MagicMock()
                    executor = ag.build_agent()

        tool_names = {t.name for t in executor.tools}
        assert tool_names == {"kubectl", "docker"}


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_kubectl(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {})
        assert "kubectl" in ag.SYSTEM_PROMPT

    def test_system_prompt_mentions_docker(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {})
        assert "docker" in ag.SYSTEM_PROMPT

    def test_system_prompt_warns_about_destructive_commands(self, monkeypatch):
        ag = _reload_agent_with_env(monkeypatch, {})
        assert "destructive" in ag.SYSTEM_PROMPT.lower()
