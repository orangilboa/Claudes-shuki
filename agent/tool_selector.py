"""
Tool Registry and Catalog

Tools are registered into categories. The planner uses the catalog to assign
tools to each subtask. The executor receives those tools directly — no LLM
selection step needed.
"""
from __future__ import annotations
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from langchain_core.tools import BaseTool

from config import config


# ── Category registry ─────────────────────────────────────────────────────────

@dataclass
class ToolCategory:
    name: str
    description: str       # shown to planner in catalog — keep short
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


def build_tool_catalog() -> str:
    """Return compact catalog for planner: '- read_file, write_file (file_write): description'."""
    _load_local_tools()
    lines = []
    for cat_name, cat in TOOL_CATEGORIES.items():
        if cat.tools:
            tools_str = ", ".join(cat.tools)
            lines.append(f"- {tools_str} ({cat_name}): {cat.description}")
    return "\n".join(lines)


# ── Local Dynamic Loading ─────────────────────────────────────────────────────

_LOCAL_TOOLS_LOADED = False

def _load_local_tools():
    """Scan .shuki/tools in workspace and CWD for extra tools."""
    global _LOCAL_TOOLS_LOADED
    if _LOCAL_TOOLS_LOADED:
        return

    roots = []
    if config.workspace.root:
        roots.append(Path(config.workspace.root))
    cwd = Path.cwd()
    if cwd not in [Path(r) for r in roots]:
        roots.append(cwd)

    for r in roots:
        tools_dir = r / ".shuki" / "tools"
        if not tools_dir.exists():
            continue

        for fp in tools_dir.glob("*.py"):
            # Skip built-ins and init
            if fp.name in ("__init__.py", "code_tools.py"):
                continue
            try:
                _load_tools_from_file(fp)
            except Exception as e:
                if config.verbose:
                    print(f"  [ToolSelector] Failed to load local tools from {fp}: {e}")

    _LOCAL_TOOLS_LOADED = True


def _load_tools_from_file(path: Path):
    """Import a python file and register all tools found within."""
    module_name = f"shuki.dynamic_tools.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools = []
    # Check for explicit list
    if hasattr(module, "ALL_TOOLS"):
        tools.extend(getattr(module, "ALL_TOOLS"))

    # Also search for decorated tools or classes
    for name, obj in inspect.getmembers(module):
        # A LangChain tool typically has these
        if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "invoke"):
            if obj not in tools:
                tools.append(obj)

    for t in tools:
        category = getattr(t, "category", "custom")
        register_tool(t, category)
        if config.verbose:
            print(f"  [ToolSelector] Registered local tool: {getattr(t, 'name', str(t))}")
