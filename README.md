# Codex — Coding Agent for Closed Networks

A Claude Code–style coding agent built with LangGraph, designed for **small-context internal LLMs** on closed Windows networks.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐     Ordered SubTask list (JSON)
│   PLANNER   │────────────────────────────────────►
└─────────────┘                                     │
                                                    ▼ (loop per task)
                                         ┌────────────────────┐
                                         │  CONTEXT ASSEMBLER │
                                         │  - task description│
                                         │  - dep. summaries  │
                                         │  - file snippets   │
                                         └────────┬───────────┘
                                                  │
                                                  ▼
                                         ┌────────────────────┐
                                         │     EXECUTOR       │◄──── tools
                                         │  Isolated session  │
                                         │  fresh context     │
                                         └────────┬───────────┘
                                                  │
                                                  ▼
                                         ┌────────────────────┐
                                         │    SUMMARIZER      │
                                         │  1-2 sentence      │
                                         │  compressed result │
                                         └────────┬───────────┘
                                                  │
                              ┌───────────────────┤
                              │                   │
                         more tasks            done
                              │                   │
                              ▼                   ▼
                          EXECUTOR          ┌────────────┐
                                            │ FINALIZER  │
                                            └─────┬──────┘
                                                  │
                                                  ▼
                                            Final Answer
```

### Why this design?

**Small context window problem:** Most self-hosted LLMs have 2K–4K token windows. A naive agent that dumps the whole codebase into context will fail immediately.

**Solution — hermetic subtasks:**
- The **Planner** breaks work into small, targeted subtasks with explicit `context_hints` (which files to look at)
- The **Context Assembler** fetches only relevant file slices and prior-task summaries, respecting a strict token budget
- Each **Executor** session is completely fresh — no accumulated conversation history
- The **Summarizer** compresses each result to 1–2 sentences, so downstream tasks pay only a tiny context cost per prior step

---

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Point at your internal LLM

Set environment variables (or edit `config.py`):

```powershell
$env:LLM_BASE_URL = "http://your-llm-server:11434"   # Ollama
$env:LLM_MODEL    = "llama3"
$env:LLM_API_KEY  = "none"   # or your key
```

Supported servers (all are OpenAI-compatible):
| Server | Base URL |
|--------|----------|
| [Ollama](https://ollama.com) | `http://localhost:11434` |
| [LM Studio](https://lmstudio.ai) | `http://localhost:1234` |
| [vLLM](https://github.com/vllm-project/vllm) | `http://server:8000` |
| [LocalAI](https://localai.io) | `http://localhost:8080` |

### 3. Configure context window

```powershell
$env:MAX_CONTEXT_TOKENS = "2048"   # adjust to your model's actual limit
```

The agent will automatically partition this budget across planner, executor, and summarizer.

### 4. Set workspace

```powershell
$env:WORKSPACE_ROOT = "C:\myproject"
```

---

## Usage

### Interactive REPL (like Claude Code)

```powershell
python main.py
```

```
codex> add docstrings to all functions in utils.py
codex> write unit tests for the DataProcessor class
codex> fix the ImportError in main.py
codex> !ls          # list workspace files
codex> !cat app.py  # read a file
codex> exit
```

### One-shot mode

```powershell
python main.py "add type hints to app.py"
python main.py --workspace C:\myproject "refactor the auth module"
python main.py --model llama3.1 --context 4096 "write a README"
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--workspace`, `-w` | Workspace directory |
| `--model`, `-m` | LLM model name |
| `--url`, `-u` | LLM base URL |
| `--context`, `-c` | Max context tokens |
| `--quiet`, `-q` | Suppress verbose output |

---

## Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file with optional line range |
| `write_file` | Write/overwrite a file |
| `patch_file` | Replace a unique string (surgical edit) |
| `create_file` | Create a new file |
| `delete_file` | Delete a file or directory |
| `list_directory` | List directory contents (with recursive option) |
| `search_in_files` | Regex search across files (like grep) |
| `run_command` | Run PowerShell/cmd commands |
| `get_file_info` | Get file size/lines without reading content |

---

## Tuning for Small Context Windows

Edit `config.py`:

```python
llm = LLMConfig(
    max_context_tokens = 2048,   # your model's actual limit

    # Token budgets per node (must sum < max_context_tokens)
    planner_budget_tokens   = 600,   # smaller = simpler plans
    executor_budget_tokens  = 1000,  # largest budget goes here
    summarizer_budget_tokens = 300,

    # Per-snippet size limits
    summary_max_chars     = 200,  # chars per prior-task summary
    file_snippet_max_chars = 500, # chars per injected file snippet

    # Max subtasks in a plan
    max_subtasks = 8,
)
```

**Rule of thumb:** `planner + executor + summarizer ≤ max_context_tokens × 0.8`

---

## Extending

### Add new tools

Add a `@tool` decorated function to `tools/code_tools.py` and register it in `ALL_TOOLS`.

### Add new node types

Add a node function to `agent/nodes.py` and wire it in `agent/graph.py`.

### Persistent sessions (resume across runs)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
saver = SqliteSaver.from_conn_string("codex.db")
graph = build_graph(checkpointer=saver)
```

### Parallel subtask execution

Replace sequential routing in `agent/nodes.py` with a `Send` API fan-out for tasks with no unmet dependencies.

---

## Project Structure

```
codex/
├── main.py              # CLI entry point / REPL
├── config.py            # All configuration
├── requirements.txt
├── agent/
│   ├── graph.py         # LangGraph graph definition
│   ├── nodes.py         # Planner, Executor, Summarizer, Finalizer
│   ├── state.py         # CodexState + SubTask dataclasses
│   ├── context.py       # Context assembler (token budget manager)
│   └── llm_client.py    # LLM client + BudgetedSession
└── tools/
    └── code_tools.py    # All LangChain tools
```
