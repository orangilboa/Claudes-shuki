"""
LangGraph nodes for the Shuki agent.

Pipeline per subtask:
  rules_selector → skill_picker → [re_planner?] → tool_selector → executor → summarizer

Global nodes:
  planner (initial + re-plan), finalizer

Node contract: receives ShukiState, returns partial update dict.
"""
from __future__ import annotations
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from config import config
from agent.state import ShukiState, SubTask
from agent.context import ContextAssembler
from agent.llm_client import BudgetedSession
from agent.rules import load_all_rules, select_relevant_rules
from agent.skills import load_all_skills, pick_skills, needs_resplit
from agent.tool_selector import select_tools_for_task, get_tools_for_names
from tools.code_tools import ALL_TOOLS, TOOL_MAP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _current_task(state: ShukiState) -> SubTask | None:
    plan = state.get("plan", [])
    idx = state.get("current_task_idx", 0)
    return plan[idx] if idx < len(plan) else None


def _list_workspace_files() -> str:
    from pathlib import Path
    ws = Path(config.workspace.root)
    if not ws.exists():
        return "(empty workspace)"
    lines = []
    for fp in sorted(ws.rglob("*")):
        if fp.is_file() and not any(p.startswith('.') for p in fp.parts):
            lines.append(str(fp.relative_to(ws)))
        if len(lines) > 40:
            lines.append("... (more files)")
            break
    return "\n".join(lines) if lines else "(empty workspace)"


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner for a coding assistant.

Break the request into an ORDERED list of small, FOCUSED subtasks.
Each subtask must target ONE type of work (reading, writing, running, documenting, etc.)
and should require AT MOST 1-2 files.

Rules:
- Max {max_subtasks} subtasks.
- ALWAYS add an explicit "read" subtask before any "write" or "edit" subtask.
  Never create a write/edit subtask without a preceding read subtask for the same file.
- Prefer small surgical changes over large rewrites.
- depends_on lists task IDs that must finish first.

Respond with ONLY valid JSON array, no markdown fences.

Format:
[
  {{
    "id": 1,
    "title": "short label",
    "description": "precise instruction for the executor",
    "depends_on": [],
    "context_hints": ["filename.py"],
    "tool_hint": "read|write|run|search|patch"
  }}
]"""


def planner_node(state: ShukiState) -> dict:
    """Initial planning: parse user request into ordered SubTask list."""
    verbose = config.verbose
    req = state["user_request"]

    session = BudgetedSession(
        system_prompt=PLANNER_SYSTEM.format(max_subtasks=config.llm.max_subtasks),
        tools=None,
        max_tokens=config.llm.planner_budget_tokens,
        verbose=verbose,
    )

    response = session.invoke(
        f"Plan this request:\n{req}\n\nWorkspace files:\n{_list_workspace_files()}"
    )

    plan = _parse_plan(str(response.content))

    if verbose:
        print(f"\n[Planner] {len(plan)} subtasks:")
        for t in plan:
            print(f"  [{t.id}] {t.title}  deps={t.depends_on}")

    return {"plan": plan, "current_task_idx": 0}


def _parse_plan(raw: str, id_offset: int = 0) -> list[SubTask]:
    """Parse JSON plan robustly, with optional id offset for inserted tasks."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return [SubTask(id=1 + id_offset, title="Execute request",
                        description=raw or "Complete the user request",
                        depends_on=[], context_hints=[])]
    try:
        data = json.loads(match.group())
        tasks = []
        for item in data:
            tasks.append(SubTask(
                id=item.get("id", len(tasks) + 1) + id_offset,
                title=item.get("title", f"Task {len(tasks)+1}"),
                description=item.get("description", ""),
                depends_on=[d + id_offset for d in item.get("depends_on", [])],
                context_hints=item.get("context_hints", []),
                tool_hint=item.get("tool_hint"),
            ))
        return tasks
    except (json.JSONDecodeError, KeyError):
        return [SubTask(id=1 + id_offset, title="Execute request",
                        description=raw, depends_on=[], context_hints=[])]


# ── Rules Selector ────────────────────────────────────────────────────────────

def rules_selector_node(state: ShukiState) -> dict:
    """Load and filter rules relevant to the current subtask."""
    task = _current_task(state)
    if task is None:
        return {}

    if config.verbose:
        print(f"\n[Rules] Selecting rules for task {task.id}: {task.title}")

    all_rules = load_all_rules()
    if not all_rules:
        if config.verbose:
            print("  [Rules] No rule files found")
        task.status = "rules"
        return {"plan": state["plan"]}

    selected = select_relevant_rules(task.description, all_rules)
    task.selected_rules = selected
    task.status = "rules"

    plan = state["plan"]
    return {"plan": plan}


# ── Skill Picker ──────────────────────────────────────────────────────────────

def skill_picker_node(state: ShukiState) -> dict:
    """
    Pick skills for the current subtask.
    Sets task.selected_skills and task.skill_prompt.
    If multiple skills match, sets task.status = 'needs_resplit'
    so the router sends us to the re-planner.
    """
    task = _current_task(state)
    if task is None:
        return {}

    if config.verbose:
        print(f"\n[Skills] Picking skills for task {task.id}: {task.title}")

    all_skills = load_all_skills()
    matched_names, merged_prompt = pick_skills(task.description, all_skills)

    task.selected_skills = matched_names
    task.skill_prompt = merged_prompt

    if needs_resplit(matched_names):
        task.status = "needs_resplit"
        if config.verbose:
            print(f"  [Skills] Multi-skill match {matched_names} → triggering re-plan")
    else:
        task.status = "skills"

    return {"plan": state["plan"]}


# ── Re-planner ────────────────────────────────────────────────────────────────

REPLANNER_SYSTEM = """You are a task splitter for a coding assistant.

A task has been identified as spanning multiple skill areas. Split it into
smaller subtasks, each targeting EXACTLY ONE skill from the list provided.
Each subtask must be independently executable.

Respond with ONLY valid JSON array, no markdown fences.

Format:
[
  {{
    "id": 1,
    "title": "short label",
    "description": "precise instruction",
    "depends_on": [],
    "context_hints": ["filename"],
    "tool_hint": "read|write|run|search|patch"
  }}
]"""


def replanner_node(state: ShukiState) -> dict:
    """
    Split the current multi-skill subtask into smaller focused subtasks.
    Inserts the new subtasks in-place in the plan and resets the index
    to point at the first new subtask.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]
    task = plan[idx]

    if verbose:
        print(f"\n[Re-planner] Splitting task {task.id}: {task.title}")
        print(f"  Skills: {task.selected_skills}")
        print(f"  Depth:  {task.replan_depth + 1}/{config.llm.max_replan_depth}")

    skills_str = "\n".join(f"- {s}" for s in task.selected_skills)

    session = BudgetedSession(
        system_prompt=REPLANNER_SYSTEM,
        tools=None,
        max_tokens=config.llm.planner_budget_tokens,
        verbose=verbose,
    )

    response = session.invoke(
        f"Original task: {task.description}\n\n"
        f"Skills matched (one per subtask):\n{skills_str}\n\n"
        f"Split into {len(task.selected_skills)} focused subtasks."
    )

    # Use a high id_offset so new task ids don't collide with existing plan
    max_id = max((t.id for t in plan), default=0)
    new_tasks = _parse_plan(str(response.content), id_offset=max_id)

    if not new_tasks:
        # Fallback: keep the original task, just clear the resplit flag
        task.status = "skills"
        return {"plan": plan}

    # Replace the multi-skill task with the new subtasks
    # Carry parent's depth so nested re-plans can still be caught
    parent_depth = task.replan_depth + 1
    for t in new_tasks:
        t.replan_depth = parent_depth
    new_plan = plan[:idx] + new_tasks + plan[idx+1:]

    if verbose:
        print(f"  [Re-planner] Replaced with {len(new_tasks)} tasks:")
        for t in new_tasks:
            print(f"    [{t.id}] {t.title}")

    return {
        "plan": new_plan,
        "current_task_idx": idx,   # restart loop at first new task
    }


# ── Tool Selector ─────────────────────────────────────────────────────────────

def tool_selector_node(state: ShukiState) -> dict:
    """Run 2-pass tool selection for the current subtask."""
    task = _current_task(state)
    if task is None:
        return {}

    if config.verbose:
        print(f"\n[ToolSelector] Selecting tools for task {task.id}: {task.title}")

    selected_names = select_tools_for_task(
        task_description=task.description,
        skill_prompt=task.skill_prompt,
        tool_hint=task.tool_hint,
    )

    # Fallback: if selector returned nothing, give all tools
    if not selected_names:
        selected_names = [t.name for t in ALL_TOOLS]
        if config.verbose:
            print("  [ToolSelector] No tools selected, falling back to all tools")

    task.selected_tool_names = selected_names
    task.status = "tools"

    if config.verbose:
        print(f"  [ToolSelector] Final tools: {selected_names}")

    return {"plan": state["plan"]}


# ── Executor ──────────────────────────────────────────────────────────────────

def _build_executor_system(task: SubTask, context_str: str) -> str:
    """
    Build a flat, simple system prompt for the executor.
    Small models do better with plain instructions than with heavily formatted sections.
    """
    parts = []

    parts.append("You are a precise assistant completing a single focused task.")
    parts.append("Rules:")
    parts.append("- Complete ONLY the task described. Do nothing else.")
    parts.append("- Before editing or writing any file, ALWAYS read it first with read_file.")
    parts.append("- Use patch_file for small edits. Only use write_file when creating from scratch.")
    parts.append("- Write real content. Never use placeholders like 'item1', 'example1', 'TODO'.")
    parts.append("- When done, write one sentence summarising exactly what you did.")

    if task.skill_prompt:
        # Only include the first 300 chars of the skill — enough for the key guidance
        skill_excerpt = task.skill_prompt[:300].strip()
        parts.append(f"\nGuidance: {skill_excerpt}")

    if task.selected_rules:
        rules_text = " | ".join(r[:120].replace("\n", " ") for r in task.selected_rules)
        parts.append(f"\nConstraints: {rules_text}")

    if context_str:
        parts.append(f"\nContext:\n{context_str}")

    return "\n".join(parts)


def executor_node(state: ShukiState) -> dict:
    """Execute the current subtask using its selected tools, skill, and rules."""
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]

    if idx >= len(plan):
        return {}

    task = plan[idx]
    task.status = "running"

    # Resolve tools
    tool_objects = get_tools_for_names(task.selected_tool_names)
    if not tool_objects:
        tool_objects = ALL_TOOLS   # safety fallback
    tool_map = {getattr(t, 'name', str(t)): t for t in tool_objects}
    tool_names_str = ", ".join(tool_map.keys())

    # Assemble context and system prompt
    assembler = ContextAssembler()
    context_str = assembler.build(task, state)
    system = _build_executor_system(task, context_str)

    if verbose:
        print(f"\n[Executor] Task {task.id}: {task.title}")
        print(f"  Tools: {tool_names_str}")
        print(f"  Skills: {task.selected_skills or ['generic']}")
        print(f"  Rules: {len(task.selected_rules)} applied")

    session = BudgetedSession(
        system_prompt=system,
        tools=tool_objects,
        max_tokens=config.llm.executor_budget_tokens,
        verbose=verbose,
    )

    MAX_TOOL_ROUNDS = 8
    response = session.invoke(f"TASK: {task.description}")

    tool_calls_made = []
    file_index_updates = {}

    for _ in range(MAX_TOOL_ROUNDS):
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tool_id = tc["id"]

            if verbose:
                print(f"  [Tool] {name}({json.dumps(args)[:80]})")

            tool_fn = tool_map.get(name)
            if tool_fn is None:
                result = f"ERROR: Tool '{name}' not available for this task"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"ERROR: {e}"

            if verbose:
                print(f"  [Result] {str(result)[:120]}")

            if name in ("write_file", "patch_file", "create_file") and "path" in args:
                file_index_updates[args["path"]] = f"Modified by task {task.id}: {task.title}"
            if name == "read_file" and "path" in args and isinstance(result, str):
                file_index_updates[args["path"]] = result[:200]

            tool_calls_made.append({"tool": name, "args": args, "result": str(result)[:300]})
            session.append_tool_result(tool_id, str(result), name)

        response = session.continue_after_tools()

    final_text = str(response.content) if response else "No response"
    task.tool_calls_made = tool_calls_made
    task.status = "done"

    if verbose:
        print(f"[Executor] Done: {final_text[:120]}")

    return {
        "task_results": [{
            "task_id": task.id,
            "title": task.title,
            "result": final_text[:500],
            "tools_used": [t["tool"] for t in tool_calls_made],
            "skills": task.selected_skills,
        }],
        "file_index": file_index_updates,
        "plan": plan,
    }


# ── Summarizer ────────────────────────────────────────────────────────────────

SUMMARIZER_SYSTEM = """Summarize what was accomplished in ONE or TWO sentences.
Be specific: mention file names, functions, and concrete changes.
No filler. Output only the summary."""


def summarizer_node(state: ShukiState) -> dict:
    """Compress executor result into a short summary. Advance task index."""
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]
    results: list[dict] = state.get("task_results", [])

    if not results or idx >= len(plan):
        return {"current_task_idx": idx + 1}

    task = plan[idx]
    last = results[-1]

    session = BudgetedSession(
        system_prompt=SUMMARIZER_SYSTEM,
        tools=None,
        max_tokens=config.llm.summarizer_budget_tokens,
        verbose=verbose,
    )
    response = session.invoke(
        f"Task: {task.description}\n"
        f"Tools: {', '.join(last.get('tools_used', [])) or 'none'}\n"
        f"Result:\n{last.get('result', '')[:600]}"
    )

    task.result_summary = str(response.content).strip()

    if verbose:
        print(f"[Summarizer] {task.result_summary}")

    return {"plan": plan, "current_task_idx": idx + 1}


# ── Finalizer ─────────────────────────────────────────────────────────────────

FINALIZER_SYSTEM = """You are a coding assistant. Write a clear, concise response
to the original user request based on the completed subtask summaries.
Mention what was created, modified, or fixed. Be helpful and specific."""


def finalizer_node(state: ShukiState) -> dict:
    """Assemble the final answer from all subtask summaries."""
    verbose = config.verbose
    plan = state.get("plan", [])
    req = state["user_request"]

    summaries = "\n".join(
        f"- {t.title}: {t.result_summary or 'completed'}"
        for t in plan
    )

    session = BudgetedSession(
        system_prompt=FINALIZER_SYSTEM,
        tools=None,
        max_tokens=config.llm.planner_budget_tokens,
        verbose=verbose,
    )
    response = session.invoke(f"User request: {req}\n\nCompleted:\n{summaries}")
    answer = str(response.content)

    if verbose:
        print(f"\n[Finalizer]\n{answer}")

    return {"final_answer": answer}


# ── Routers ───────────────────────────────────────────────────────────────────

def route_after_skill_picker(state: ShukiState) -> str:
    """After skill picking: re-plan if multi-skill (and depth allows), else select rules."""
    task = _current_task(state)
    if task and task.status == "needs_resplit":
        if task.replan_depth < config.llm.max_replan_depth:
            return "replan"
        else:
            if config.verbose:
                print(f"  [Router] Re-plan depth limit reached for task {task.id}, proceeding")
            task.status = "skills"
    return "select_rules"


def route_after_summarizer(state: ShukiState) -> str:
    """After summarizing: keep looping or finalize."""
    plan = state.get("plan", [])
    idx = state.get("current_task_idx", 0)
    if idx >= len(plan):
        return "finalize"
    # Check dependencies satisfied
    current = plan[idx]
    for dep_id in current.depends_on:
        dep = next((t for t in plan if t.id == dep_id), None)
        if dep and dep.status != "done":
            return "finalize"
    return "continue"
