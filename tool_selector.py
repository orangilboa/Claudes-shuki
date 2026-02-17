"""
Two-Pass Tool Selector (with single-pass fallback for small pools)

Pool size ≤ 30 tools  →  Single pass: LLM sees all tools, picks directly.
Pool size  > 30 tools  →  Two passes:

  Pass 1: LLM sees only CATEGORY names + one-line descriptions.
           It picks which categories apply to this subtask.
           Cost: tiny — O(num_categories), not O(num_tools).

  Pass 2: LLM sees only the tools WITHIN the selected categories.
           It picks the specific tools to give the executor.
           Cost: bounded — only the subset, never the full pool.

The threshold is configurable via TWO_PASS_THRESHOLD (default: 30).
This design scales to 200+ tools without ever blowing the context window.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Any

from langchain_core.tools import BaseTool

from config import config
from agent.llm_client import BudgetedSession


# ── Category registry ─────────────────────────────────────────────────────────

@dataclass
class ToolCategory:
    name: str
    description: str       # shown to LLM in pass 1 — keep short
    tools: list[str] = field(default_factory=list)   # tool names in this category


# Master category list — extend this as the tool pool grows
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "file_read": ToolCategory(
        name="file_read",
        description="Reading files, getting file metadata, listing directories",
        tools=["read_file", "get_file_info", "list_directory"],
    ),
    "file_write": ToolCategory(
        name="file_write",
        description="Creating, writing, patching, or deleting files",
        tools=["write_file", "patch_file", "create_file", "delete_file"],
    ),
    "code_search": ToolCategory(
        name="code_search",
        description="Searching for patterns, symbols, or text across files",
        tools=["search_in_files"],
    ),
    "shell": ToolCategory(
        name="shell",
        description="Running shell commands, scripts, tests, build tools, git",
        tools=["run_command"],
    ),
    # ── Future categories (tools not yet implemented) ──────────────────────
    # "web":        http_get, http_post, download_file
    # "database":   sql_query, db_schema
    # "git":        git_diff, git_log, git_blame
    # "clipboard":  read_clipboard, write_clipboard
    # "browser":    open_url, screenshot
}


# ── Dynamic tool registry ─────────────────────────────────────────────────────
# Maps tool_name → (BaseTool or callable, category_name)
_TOOL_REGISTRY: dict[str, tuple[Any, str]] = {}


def register_tool(tool: Any, category: str, name: Optional[str] = None) -> None:
    """
    Register a tool (BaseTool, plain callable, or any object with .name)
    into a category. Call this at startup for external/dynamic tools.

    Example:
        register_tool(my_api_tool, "web")
        register_tool(mcp_tool, "database", name="run_sql")
    """
    tool_name = name or getattr(tool, "name", str(tool))
    _TOOL_REGISTRY[tool_name] = (tool, category)
    if category in TOOL_CATEGORIES:
        if tool_name not in TOOL_CATEGORIES[category].tools:
            TOOL_CATEGORIES[category].tools.append(tool_name)
    else:
        TOOL_CATEGORIES[category] = ToolCategory(
            name=category,
            description=f"Custom category: {category}",
            tools=[tool_name],
        )


def get_tool_object(name: str) -> Optional[Any]:
    """Retrieve a registered tool by name."""
    if name in _TOOL_REGISTRY:
        return _TOOL_REGISTRY[name][0]
    # Fall back to the built-in code_tools TOOL_MAP
    from tools.code_tools import TOOL_MAP
    return TOOL_MAP.get(name)


def get_tools_for_names(names: list[str]) -> list[Any]:
    """Resolve a list of tool names to tool objects, skipping unknowns."""
    tools = []
    for name in names:
        obj = get_tool_object(name)
        if obj is not None:
            tools.append(obj)
        elif config.verbose:
            print(f"  [ToolSelector] Warning: unknown tool '{name}' skipped")
    return tools


# ── Pass 1: category selection ────────────────────────────────────────────────

_CAT_SELECTOR_SYSTEM = """You are a tool category selector for a coding assistant.

Given a task, select the categories of tools that will be needed.
Respond with ONLY a comma-separated list of category names, e.g.: file_read,shell
If truly no tools are needed, respond with: none
No explanation, no other text."""


def _select_categories(
    task_description: str,
    skill_prompt: str,
    tool_hint: Optional[str],
) -> list[str]:
    """Pass 1: pick which tool categories apply."""

    # Build compact category index
    cat_lines = [f"- {cat}: {info.description}"
                 for cat, info in TOOL_CATEGORIES.items()
                 if info.tools]   # only show categories with registered tools
    cat_index = "\n".join(cat_lines)

    hint_note = f"\nPlanner hint: {tool_hint}" if tool_hint else ""
    skill_note = ""
    if skill_prompt:
        # Trim skill to first 100 chars for category selector context
        skill_note = f"\nSkill context: {skill_prompt[:100]}"

    prompt = (
        f"Task: {task_description[:300]}"
        f"{hint_note}{skill_note}\n\n"
        f"Available categories:\n{cat_index}"
    )

    session = BudgetedSession(
        system_prompt=_CAT_SELECTOR_SYSTEM,
        tools=None,
        max_tokens=60,
        verbose=config.verbose,
    )

    try:
        response = session.invoke(prompt)
        raw = str(response.content).strip().lower()
        if raw == "none":
            return []
        valid = set(TOOL_CATEGORIES.keys())
        selected = [t.strip() for t in re.split(r"[,\s]+", raw)
                    if t.strip() in valid]
        if config.verbose:
            print(f"  [ToolSelector P1] Categories: {selected}")
        return selected
    except Exception as e:
        if config.verbose:
            print(f"  [ToolSelector P1] Failed: {e}")
        return list(TOOL_CATEGORIES.keys())   # fallback: all categories


# ── Pass 2: specific tool selection ──────────────────────────────────────────

_TOOL_SELECTOR_SYSTEM = """You are a tool selector for a coding assistant.

Given a task and a list of available tools, select ONLY the tools that are
actually needed for this specific task. Prefer fewer tools — only pick what
the executor will likely call.

Respond with ONLY a comma-separated list of tool names, e.g.: read_file,run_command
No explanation, no other text."""


def _select_specific_tools(
    task_description: str,
    candidate_tools: list[str],
    tool_descriptions: dict[str, str],
) -> list[str]:
    """Pass 2: pick specific tools from the filtered candidate list."""
    if not candidate_tools:
        return []

    tool_lines = [f"- {name}: {tool_descriptions.get(name, '(no description)')}"
                  for name in candidate_tools]
    tool_index = "\n".join(tool_lines)

    prompt = (
        f"Task: {task_description[:300]}\n\n"
        f"Available tools:\n{tool_index}"
    )

    session = BudgetedSession(
        system_prompt=_TOOL_SELECTOR_SYSTEM,
        tools=None,
        max_tokens=80,
        verbose=config.verbose,
    )

    try:
        response = session.invoke(prompt)
        raw = str(response.content).strip().lower()
        valid = set(candidate_tools)
        selected = [t.strip() for t in re.split(r"[,\s]+", raw)
                    if t.strip() in valid]
        if config.verbose:
            print(f"  [ToolSelector P2] Tools: {selected}")
        return selected if selected else candidate_tools  # fallback: all candidates
    except Exception as e:
        if config.verbose:
            print(f"  [ToolSelector P2] Failed: {e}")
        return candidate_tools


# ── Public entry point ────────────────────────────────────────────────────────

# Below this pool size the category pass is skipped — the LLM can see all
# tools directly in a single pass without blowing the context window.
TWO_PASS_THRESHOLD = 30


def _all_registered_tool_names() -> list[str]:
    """Return every tool name known to the system (categories + registry)."""
    seen: set[str] = set()
    names: list[str] = []
    for cat in TOOL_CATEGORIES.values():
        for t in cat.tools:
            if t not in seen:
                names.append(t)
                seen.add(t)
    for t in _TOOL_REGISTRY:
        if t not in seen:
            names.append(t)
            seen.add(t)
    return names


def select_tools_for_task(
    task_description: str,
    skill_prompt: str = "",
    tool_hint: Optional[str] = None,
    tool_descriptions: Optional[dict[str, str]] = None,
) -> list[str]:
    """
    Return the tool names the executor should receive for this subtask.

    Pool size ≤ TWO_PASS_THRESHOLD (30):
        Single pass — LLM sees all tools directly. No category overhead.

    Pool size > TWO_PASS_THRESHOLD:
        Pass 1 — LLM picks categories (tiny context cost, O(num_categories)).
        Pass 2 — LLM picks specific tools within those categories.

    tool_descriptions: optional {name: description} override.
    """
    if tool_descriptions is None:
        tool_descriptions = _build_tool_descriptions()

    all_tools = _all_registered_tool_names()
    pool_size = len(all_tools)

    if config.verbose:
        print(f"  [ToolSelector] Pool size: {pool_size} "
              f"({'single-pass' if pool_size <= TWO_PASS_THRESHOLD else '2-pass'})")

    if pool_size <= TWO_PASS_THRESHOLD:
        # ── Single pass: show every tool, pick directly ────────────────────────
        return _select_specific_tools(task_description, all_tools, tool_descriptions)

    # ── Two-pass: categories first, then specific tools ────────────────────────
    selected_cats = _select_categories(task_description, skill_prompt, tool_hint)
    if not selected_cats:
        return []

    candidate_tools: list[str] = []
    seen: set[str] = set()
    for cat in selected_cats:
        for tool_name in TOOL_CATEGORIES[cat].tools:
            if tool_name not in seen:
                candidate_tools.append(tool_name)
                seen.add(tool_name)

    if not candidate_tools:
        return []

    return _select_specific_tools(task_description, candidate_tools, tool_descriptions)


def _build_tool_descriptions() -> dict[str, str]:
    """Build {tool_name: one-line description} from all registered tools."""
    desc: dict[str, str] = {}
    from tools.code_tools import TOOL_MAP
    for name, tool in TOOL_MAP.items():
        doc = getattr(tool, "description", None) or ""
        # Use first sentence only
        first_sentence = doc.split(".")[0].strip()[:100]
        desc[name] = first_sentence
    for name, (tool, _) in _TOOL_REGISTRY.items():
        doc = getattr(tool, "description", None) or getattr(tool, "__doc__", "") or ""
        first_sentence = doc.split(".")[0].strip()[:100]
        desc[name] = first_sentence
    return desc
