# Shuki — AI Assistant for Closed Networks

## Project Purpose

Shuki is a general-purpose AI assistant designed for deployment on internal, air-gapped, or otherwise closed networks. It connects to a locally-hosted LLM (e.g. via Ollama) instead of cloud APIs, making it suitable for environments with strict network isolation or data-sensitivity requirements.

Primary use cases:
- **Coding assistance** — reading, writing, editing, and refactoring code inside a sandboxed workspace
- **Email triage** — sorting, summarising, and drafting responses to emails
- **Information retrieval** — finding and synthesising information from internal documents and files
- **Meeting setup and coordination** — looking at calendars, chat with invitees to find the best suitable time

---

## Architecture

The agent is built on **LangGraph** and follows a multi-node pipeline per subtask:

```
discovery → planner → skill_picker → [replanner if multi-skill]
                                   → rules_selector
                                   → tool_selector
                                   → reasoner (read-only, produces edit plan)
                                   → writer   (applies edits mechanically)
                                   → verifier → [retry reasoner on fail]
                                   → summarizer → [loop or finalize]
                                   → finalizer
```

Key source files:

| File | Role |
|---|---|
| [main.py](main.py) | Entry point — interactive REPL and one-shot mode |
| [config.py](config.py) | All configuration (LLM endpoint, token budgets, workspace path) |
| [agent/graph.py](agent/graph.py) | LangGraph state machine definition |
| [agent/nodes.py](agent/nodes.py) | All node implementations |
| [agent/state.py](agent/state.py) | `ShukiState` TypedDict |
| [agent/context.py](agent/context.py) | Context assembly and token-budget management |
| [agent/skills.py](agent/skills.py) | Skill discovery and selection |
| [agent/rules.py](agent/rules.py) | Rules loader |
| [agent/tool_selector.py](agent/tool_selector.py) | Tool selection logic |
| [agent/llm_client.py](agent/llm_client.py) | `BudgetedSession` — token-aware LLM wrapper |

---

## Configuration

All settings live in [config.py](config.py) and can be overridden via environment variables or `.env`:

| Env var | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434` | Local LLM endpoint (Ollama or compatible) |
| `LLM_MODEL` | `llama3` | Model name |
| `LLM_API_KEY` | `ollama` | API key (if the endpoint requires one) |
| `MAX_INPUT_TOKENS` | `8192` | Input context budget |
| `MAX_OUTPUT_TOKENS` | `4096` | Max output tokens per LLM call |
| `WORKSPACE_ROOT` | `./workspace` | Sandboxed working directory for the agent |
| `SHUKI_SHELL` | `powershell` | Shell for command execution (Windows) |
| `SHUKI_VERBOSE` | `1` | Enable debug logging |

---

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (First run only) set up folder structure
python setup.py

# 3. Configure your local LLM
export LLM_BASE_URL="http://your-server:11434"
export LLM_MODEL="llama3"

# 4. Run interactive REPL
python main.py

# 5. Or one-shot mode
python main.py "add type hints to app.py"
```

---

## Skills & Rules

Skills and rules live under `.shuki/` and are plain markdown files:

- `.shuki/skill/` — task-type prompts (e.g. `coding.md`, `email.md`, `documentation.md`)
- `.shuki/rules/` — coding constraints applied to every task (e.g. `always_type_hints.md`)

Add new skills or rules by dropping a `.md` file into the relevant directory. The agent discovers them automatically at runtime.

---

## Development Notes

- The project targets **Python 3.10+** and runs on both Windows and macOS/Linux.
- The agent is intentionally token-budget-aware; input context is split 20% / 60% / 20% across planner, executor, and summariser nodes.
- Tools are sandboxed to `WORKSPACE_ROOT`. Do not expand tool access outside this boundary without careful review.
- Secrets (API keys, `.env`) are git-ignored. Never commit `.env` or `.key`.