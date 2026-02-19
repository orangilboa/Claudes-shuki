# Shuki — Coding Agent for Closed Networks

A Claude Code–style coding agent built with LangGraph, designed for **small-context internal LLMs** on closed Windows networks.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│   PLANNER   │  Breaks request into ordered subtasks (one focused job each)
└──────┬──────┘
       │  for each subtask:
       ▼
┌─────────────────┐
│ RULES SELECTOR  │  Loads .shuki rules from global + project dirs
│                 │  LLM picks only rules relevant to THIS subtask
└──────┬──────────┘
       ▼
┌─────────────────┐
│  SKILL PICKER   │  Picks skill(s) from bundled + user skills
└──────┬──────────┘
       │
   [multi-skill?]──YES──► REPLANNER ──► back to RULES SELECTOR (new subtasks)
       │ NO
       ▼
┌─────────────────────────┐
│    TOOL SELECTOR        │
│  Pass 1: pick categories│  LLM sees only category names (~5-10 options)
│  Pass 2: pick tools     │  LLM sees only tools in selected categories
└──────┬──────────────────┘
       ▼
┌─────────────────┐
│    EXECUTOR     │  Isolated session: task + skill prompt + rules + selected tools
└──────┬──────────┘
       ▼
┌─────────────────┐
│   SUMMARIZER    │  Compresses result to 1-2 sentences for downstream context
└──────┬──────────┘
       │
  [more tasks?]──YES──► back to RULES SELECTOR
       │ NO
       ▼
┌─────────────────┐
│   FINALIZER     │  Assembles final answer from all summaries
└─────────────────┘
```

### Why this design handles small context windows

Every executor session is hermetically sealed:
- **Rules**: LLM picks only relevant rules (not all rules)
- **Skills**: Matched skill provides targeted instructions
- **Tools**: LLM sees only the 2-4 tools needed (not all 20+)  
- **Context**: Prior-task summaries (1-2 sentences each, not full results)
- **Files**: Snippets only, not whole files

Each pipeline step uses a *tiny* LLM call (60-100 token output) so the
executor session gets the maximum remaining budget.

---

## Setup

### 1. Install

```powershell
pip install -r requirements.txt
```

### 2. Configure your LLM endpoint

```powershell
$env:LLM_BASE_URL = "http://your-server:11434"   # Ollama, LM Studio, vLLM, etc.
$env:LLM_MODEL    = "llama3"
$env:LLM_API_KEY  = "none"
$env:MAX_INPUT_TOKENS = "8192"             # input context budget
$env:MAX_OUTPUT_TOKENS = "4096"            # your model's output limit
$env:WORKSPACE_ROOT = "C:\myproject"
```

### 3. Add your rules (optional but recommended)

Copy `.shuki.example/` files to configure your team's coding standards:

```powershell
# Global rules (apply to all projects)
mkdir "$env:LOCALAPPDATA\.shuki\rules"
copy .shuki.example\* "$env:LOCALAPPDATA\.shuki\rules\"

# Project-local rules (checked into your repo)
mkdir .shuki
# Add .md or .txt files — one rule per file
```

### 4. Add custom skills (optional)

```powershell
mkdir "$env:LOCALAPPDATA\.shuki\skills"
# Add .md or .txt skill files here
```

---

## Usage

### Interactive REPL

```powershell
python main.py
```

```
shuki> add type hints to all functions in auth.py
shuki> write pytest tests for the DataProcessor class
shuki> fix the ImportError in main.py and update the README
shuki> !ls          # list workspace files
shuki> !cat app.py  # read a file
shuki> exit
```

### One-shot

```powershell
python main.py "refactor the auth module to use dataclasses"
python main.py --workspace C:\myproject --model llama3.1 --context 4096 "write a README"
```

---

## Rules (`.shuki` files)

Rules constrain **how** tasks are done — coding style, forbidden patterns, conventions.

**File locations** (both are loaded and merged):
```
%LOCALAPPDATA%\.shuki\rules\    ← your personal rules (all projects)
{workspace}\.shuki\             ← project-specific rules
```

**Format:** One `.md` or `.txt` file per rule. The filename becomes the rule name.

```markdown
# No print statements in production code

Never add print() or console.log() to production files.
Use the project logger: logging.getLogger(__name__).info(...)
```

The rules selector runs a small LLM call per subtask that picks only rules
relevant to what that subtask is actually doing. A "no print statements" rule
won't be injected into a task that's only reading file metadata.

---

## Skills (`skills/` directory)

Skills define **how to approach** a category of work. They're injected as a
focused system prompt into the executor for matching tasks.

**Bundled skills** (in `skills/`):
- `coding.md` — general software development
- `testing.md` — writing and running tests
- `documentation.md` — docstrings, READMEs, comments
- `devops.md` — scripts, CI/CD, configuration

**User skills** (`%LOCALAPPDATA%\.shuki\skills\`): add your own.

**Skill file format:**
```markdown
# Skill Name — Short Description

You are a [role] completing a [type] task.

## Approach
...instructions...

## Preferred tools
tool_a, tool_b, tool_c
```

**Multi-skill re-planning:** when a subtask matches 2+ skills, it's automatically
split into smaller subtasks — one per skill — before execution. This prevents
mixed-skill tasks from degrading quality on small-context models.

---

## Tools

### Bundled tools

| Tool | Category | Description |
|------|----------|-------------|
| `read_file` | file_read | Read with optional line range |
| `get_file_info` | file_read | Size/line count without reading |
| `list_directory` | file_read | Directory tree listing |
| `write_file` | file_write | Create or overwrite a file |
| `patch_file` | file_write | Replace a unique string (surgical edit) |
| `create_file` | file_write | Create new file (fails if exists) |
| `delete_file` | file_write | Delete file or directory |
| `search_in_files` | code_search | Regex/literal search across files |
| `run_command` | shell | Run PowerShell/cmd commands |

### Tool categories

The selector auto-adapts based on pool size:

```
≤ 30 tools  →  single pass: LLM sees all tools, picks directly (no overhead)
 > 30 tools  →  two passes:
                  Pass 1: LLM sees only category names (~5-10 options)
                  Pass 2: LLM sees only tools within selected categories
```

### Registering external tools

```python
from agent.tool_selector import register_tool

# Any LangChain tool, MCP tool, or callable
register_tool(my_http_tool, category="web")
register_tool(my_db_tool,   category="database", name="run_sql")
```

Registered tools appear in category selection and Pass 2 automatically.

---

## Adding New Tool Categories

In `agent/tool_selector.py`:

```python
TOOL_CATEGORIES["git"] = ToolCategory(
    name="git",
    description="Git operations: diff, log, blame, commit, branch",
    tools=["git_diff", "git_log", "git_commit"],
)
```

Then implement the tools in `tools/` and call `register_tool()` at startup.

---

## Configuration Reference

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:11434` | LLM server URL |
| `LLM_MODEL` | `llama3` | Model name |
| `LLM_API_KEY` | `ollama` | API key |
| `MAX_INPUT_TOKENS` | `8192` | Input context budget |
| `MAX_OUTPUT_TOKENS` | `4096` | Max output tokens per API call |
| `WORKSPACE_ROOT` | `./workspace` | Working directory |
| `SHUKI_SHELL` | `powershell` | Shell for run_command |
| `SHUKI_VERBOSE` | `1` | Print debug info |
| `SHUKI_GLOBAL_DIR` | auto | Override global .shuki location |
| `SHUKI_SKILLS_DIR` | auto | Override bundled skills location |
| `COMMAND_TIMEOUT` | `30` | Shell command timeout (seconds) |

---

## Project Structure

```
shuki/
├── main.py                  # CLI entry point / REPL
├── config.py                # All configuration
├── requirements.txt
│
├── agent/
│   ├── graph.py             # LangGraph wiring
│   ├── nodes.py             # All node functions + routers
│   ├── state.py             # ShukiState + SubTask dataclasses
│   ├── context.py           # Context assembler (token budget manager)
│   ├── llm_client.py        # BudgetedSession (isolated LLM calls)
│   ├── rules.py             # Rules loader + relevance selector
│   ├── skills.py            # Skill loader + picker
│   └── tool_selector.py     # 2-pass tool category + specific selection
│
├── tools/
│   └── code_tools.py        # Built-in LangChain tools
│
├── skills/                  # Bundled skill definitions (md/txt)
│   ├── coding.md
│   ├── testing.md
│   ├── documentation.md
│   └── devops.md
│
└── .shuki.example/          # Example rule files to copy to your .shuki dir
    ├── no_print_statements.md
    ├── always_type_hints.md
    └── prefer_pathlib.md
```
