"""
Configuration for the Codex agent.
Designed for small-context internal LLMs on closed networks.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    # --- Internal LLM endpoint ---
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")  # e.g. Ollama
    model: str = os.getenv("LLM_MODEL", "llama3")
    api_key: str = os.getenv("LLM_API_KEY", "ollama")  # some endpoints require a key

    # --- Context window budget ---
    # Leave room for the model's output. Set this conservatively.
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))

    # Rough chars-per-token estimate (conservative for code)
    chars_per_token: float = 3.5

    # Per-node token budgets — derived in __post_init__ from max_context_tokens:
    #   planner: 20%, executor: 60%, summarizer: 20%
    planner_budget_tokens: int = 0
    executor_budget_tokens: int = 0
    summarizer_budget_tokens: int = 0

    def __post_init__(self):
        self.planner_budget_tokens    = int(self.max_context_tokens * 0.20)
        self.executor_budget_tokens   = int(self.max_context_tokens * 0.60)
        self.summarizer_budget_tokens = int(self.max_context_tokens * 0.20)

    # Max tokens per prior-task summary injected into executor context
    summary_max_chars: int = 300

    # Max chars to include from any single file snippet
    file_snippet_max_chars: int = 600

    # Max subtasks the planner may generate
    max_subtasks: int = 12


@dataclass
class WorkspaceConfig:
    root: str = os.getenv("WORKSPACE_ROOT", "./workspace")
    # Shell to use for commands (Windows)
    shell: str = os.getenv("CODEX_SHELL", "powershell")
    # Extra allowed paths (read-only access outside workspace)
    allowed_read_paths: list = field(default_factory=list)
    # Command timeout in seconds
    command_timeout: int = int(os.getenv("COMMAND_TIMEOUT", "30"))


@dataclass
class PathsConfig:
    root: Path = Path(__file__).parent
    shuki: Path = root / ".shuki"
    skill: Path = shuki / "skill"
    rules: Path = shuki / "rules"
    tools: Path = shuki / "tools"

@dataclass 
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    verbose: bool = os.getenv("CODEX_VERBOSE", "1") == "1"


# Singleton
config = Config()
