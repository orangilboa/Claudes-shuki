"""
LangGraph nodes for the Shuki agent.

Pipeline per subtask:
  planner (picks skill + tools per subtask)
    → executor (ReAct loop: reads then writes directly)
    → verifier (confirms file changes landed)
    → summarizer

Global nodes: discovery, finalizer
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage

from config import config
from agent.state import ShukiState, SubTask
from agent.context import ContextAssembler
from agent.llm_client import BudgetedSession
from agent.rules import load_all_rules, format_all_rules
from agent.skills import load_all_skills, get_skill_content, build_skills_catalog
from agent.tool_selector import build_tool_catalog, get_tools_for_names
from tools.code_tools import ALL_TOOLS, TOOL_MAP, READ_TOOLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _current_task(state: ShukiState) -> SubTask | None:
    plan = state.get("plan", [])
    idx  = state.get("current_task_idx", 0)
    return plan[idx] if idx < len(plan) else None


def _needs_discovery(request: str) -> bool:
    """Heuristic to check if a request likely needs workspace file discovery."""
    req = request.lower()
    # Mentioning common file extensions or relative/absolute path separators
    if re.search(r"\.[a-z0-9]{1,4}\b|/|\\", req):
        return True
    # Action keywords that imply editing or exploring existing code
    actions = ["fix", "refactor", "update", "modify", "patch", "change", "add to", "integrate", "debug", "analyze"]
    if any(a in req for a in actions):
        return True
    # Keywords that imply repo-wide knowledge
    concepts = ["repo", "workspace", "codebase", "project", "module", "function", "class", "logic"]
    if any(c in req for c in concepts):
        return True
    return False


# ── Discovery ──────────────────────────────────────────────────────────────────

DISCOVERY_SYSTEM = """You are a file discovery agent.
Explore the workspace to find ALL files and information relevant to the user request.

ALWAYS start with: list_directory(path=".", recursive=true) to see the full project structure.
Then use search_in_files to find the relevant code across ALL subdirectories.

When searching for patterns (e.g. print statements), search the ENTIRE workspace:
  search_in_files(pattern="print\\(", path=".", file_glob="*.py")
This searches recursively through all subdirectories including agent/, tools/, etc.

Provide a complete summary: list every file that contains relevant code, with line counts.
"""

def discovery_node(state: ShukiState) -> dict:
    verbose = config.verbose
    if verbose:
        print("\n[Discovery] Exploring workspace...")

    # We use the read tools to explore
    read_tools = [TOOL_MAP[n] for n in READ_TOOLS if n in TOOL_MAP]
    read_tool_map = {t.name: t for t in read_tools}

    session = BudgetedSession(
        system_prompt=DISCOVERY_SYSTEM,
        tools=read_tools,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )

    prompt = f"Find relevant files for this request:\n{state['user_request']}"
    if state.get("discovery_results"):
        prompt += f"\n\nPrevious discovery status:\n{state['discovery_results']}"

    response = session.invoke(prompt)

    # Tool loop for discovery
    MAX_ROUNDS = 5
    for _ in range(MAX_ROUNDS):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid  = tc["id"]
            if verbose:
                print(f"  [Discovery] {name}({json.dumps(args)})")
            tool_fn = read_tool_map.get(name)
            result = tool_fn.invoke(args) if tool_fn else f"ERROR: tool '{name}' not available"
            session.append_tool_result(tid, str(result), name)
        response = session.continue_after_tools()

    discovery_results = str(response.content)
    if verbose:
        print(f"[Discovery] Summary: {discovery_results[:300].replace(chr(10), ' ')}...")

    return {"discovery_results": discovery_results}


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_TEMPLATE = """You are a task planner for a general-purpose assistant.

Use the provided discovery results to break the user request into an ORDERED list of small, FOCUSED subtasks.
Each subtask MUST touch exactly ONE file or resource.

For each subtask, assign the most appropriate skill and the minimal set of tools needed.

Available skills:
{skills_catalog}

Available tools:
{tools_catalog}

If the discovery results are insufficient and you need more information to make a solid plan,
respond with a JSON object asking for more discovery:
{{
  "action": "search",
  "queries": ["what you need to find", "another query"]
}}

Otherwise, respond with a JSON array of subtasks:
[
  {{
    "id": 1,
    "title": "short label",
    "description": "precise instruction for ONE file",
    "depends_on": [],
    "context_hints": ["filename"],
    "skill": "coding",
    "tools": ["read_file", "write_file"]
  }}
]

Respond with ONLY valid JSON, no markdown fences.
"""


def planner_node(state: ShukiState) -> dict:
    verbose = config.verbose

    # Load skills and tools catalogs for the system prompt
    all_skills = load_all_skills()
    skills_catalog = build_skills_catalog(all_skills)
    tools_catalog = build_tool_catalog()

    system = PLANNER_SYSTEM_TEMPLATE.format(
        skills_catalog=skills_catalog,
        tools_catalog=tools_catalog,
    )

    session = BudgetedSession(
        system_prompt=system,
        tools=None,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )
    prompt = (
        f"Plan this request:\n{state['user_request']}\n\n"
        f"Discovery results:\n{state['discovery_results']}"
    )
    response = session.invoke(prompt)
    raw = str(response.content)

    # Check if planner wants more search
    search_req = _parse_search_request(raw)
    if search_req:
        if verbose:
            print(f"\n[Planner] Needs more info: {search_req.get('queries')}")
        return {
            "discovery_results": f"PLANNER NEEDS MORE INFO: {json.dumps(search_req)}",
            "plan": []  # Signal router to loop back
        }

    plan = _parse_plan(raw)
    if verbose:
        print(f"\n[Planner] {len(plan)} subtasks:")
        for t in plan:
            print(f"  [{t.id}] {t.title}  skill={t.skill}  tools={t.tools}  deps={t.depends_on}")
    return {"plan": plan, "current_task_idx": 0}


def _parse_search_request(raw: str) -> Optional[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Try to find a JSON object with "action": "search"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and data.get("action") == "search":
                return data
        except json.JSONDecodeError:
            pass
    return None


def _parse_plan(raw: str, id_offset: int = 0) -> list[SubTask]:
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
                skill=item.get("skill", "generic"),
                tools=item.get("tools", []),
            ))
        return tasks
    except (json.JSONDecodeError, KeyError):
        return [SubTask(id=1 + id_offset, title="Execute request",
                        description=raw, depends_on=[], context_hints=[])]


# ── Executor ──────────────────────────────────────────────────────────────────

EXECUTOR_SYSTEM = """You are a precise agent completing one focused task.

Use your tools to read, then modify files directly.

Guidelines:
- Always read_file before modifying to see current content
- For 1-3 surgical changes: patch_file with exact verbatim text from the file
- For bulk changes (4+ occurrences): write_file with complete modified content,
  OR run_command with a python/sed one-liner for reliability
- After writing: read_file to confirm the change landed
- Do not explain your plan — just use tools and complete the task
{skill_section}{rules_section}"""


def executor_node(state: ShukiState) -> dict:
    """
    ReAct executor: reads then writes directly. Replaces reasoner + writer.
    Runs up to 15 tool-call rounds per subtask.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]
    if idx >= len(plan):
        return {}

    task = plan[idx]
    task.status = "running"

    if verbose:
        print(f"\n[Executor] Task {task.id}: {task.title}  skill={task.skill}  tools={task.tools}")

    # Load skills and rules
    all_skills = load_all_skills()
    all_rules = load_all_rules()
    skill_content = get_skill_content(task.skill, all_skills)
    rules_content = format_all_rules(all_rules)

    skill_section = f"\n\n## Skill: {task.skill}\n{skill_content}" if skill_content else ""
    rules_section = f"\n\n{rules_content}" if rules_content else ""

    system = EXECUTOR_SYSTEM.format(
        skill_section=skill_section,
        rules_section=rules_section,
    )

    # Resolve tools — use planner-assigned list, fall back to all tools
    if task.tools:
        tool_objects = get_tools_for_names(task.tools)
        if not tool_objects:
            tool_objects = list(ALL_TOOLS)
    else:
        tool_objects = list(ALL_TOOLS)

    tool_map = {t.name: t for t in tool_objects}

    # Build context from prior task outputs and file index
    assembler = ContextAssembler()
    context_str = assembler.build(task, state)

    # On retry: inject previous executor output and the error for context
    retry_ctx = ""
    if task.retry_count > 0:
        retry_ctx = (
            f"\n\n━━━ PREVIOUS ATTEMPT FAILED ━━━\n"
            f"Verification failed: {task.verify_message}\n"
            f"Previous output:\n{task.executor_output[:600]}\n"
            f"━━━ END PREVIOUS ATTEMPT ━━━\n\n"
            f"IMPORTANT: Please correct the issue and complete the task."
        )

    prompt = f"TASK: {task.description}"
    if context_str:
        prompt += f"\n\nContext:\n{context_str}"
    if retry_ctx:
        prompt += retry_ctx

    session = BudgetedSession(
        system_prompt=system,
        tools=tool_objects,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )

    response = session.invoke(prompt)
    file_index_updates: dict[str, str] = {}
    files_modified: list[str] = []

    # ReAct loop — up to 15 tool rounds
    MAX_ROUNDS = 15
    for _ in range(MAX_ROUNDS):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid  = tc["id"]
            if verbose:
                print(f"  [Executor] {name}({json.dumps(args)})")
            tool_fn = tool_map.get(name)
            if tool_fn is None:
                # Try global tool map as fallback
                tool_fn = TOOL_MAP.get(name)
            if tool_fn is None:
                result = f"ERROR: tool '{name}' not available"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"ERROR: tool call failed — {e}"
            if verbose:
                print(f"  [Result] {str(result)[:200]}")
            # Track write/patch operations for verifier
            if name in ("write_file", "patch_file", "create_file") and "path" in args:
                fp = args["path"]
                if str(result).startswith("OK") and fp not in files_modified:
                    files_modified.append(fp)
                    try:
                        file_index_updates[fp] = TOOL_MAP["read_file"].invoke({"path": fp})
                    except Exception:
                        file_index_updates[fp] = f"Modified by task {task.id}"
            elif name == "read_file" and "path" in args and isinstance(result, str):
                file_index_updates[args["path"]] = result
            session.append_tool_result(tid, str(result), name)
        response = session.continue_after_tools()

    task.executor_output = str(response.content)
    task.files_modified = files_modified
    task.status = "done"

    return {"plan": plan, "file_index": file_index_updates}


# ── Verifier ──────────────────────────────────────────────────────────────────

def verifier_node(state: ShukiState) -> dict:
    """
    Confirm that the executor's file changes actually landed.
    Sets task.verify_passed and task.verify_message.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx:  int           = state["current_task_idx"]
    if idx >= len(plan):
        return {}

    task = plan[idx]

    if verbose:
        print(f"\n[Verifier] Task {task.id}")

    # No file changes attempted — treat as passed (read-only or info task)
    if not task.files_modified:
        task.verify_passed  = True
        task.verify_message = "No file changes — read-only or informational task."
        task.status = "done"
        return {"plan": plan}

    # Verify each modified file exists and is non-empty
    failed_files = []
    file_index_updates: dict[str, str] = {}
    for fp in task.files_modified:
        try:
            content = TOOL_MAP["read_file"].invoke({"path": fp})
            content_str = str(content)
            if content_str.startswith("ERROR") or not content_str.strip():
                failed_files.append(fp)
            else:
                file_index_updates[fp] = content_str[:400]
        except Exception:
            failed_files.append(fp)

    if failed_files:
        task.verify_passed  = False
        task.verify_message = f"FAIL: could not verify files: {failed_files}"
    else:
        task.verify_passed  = True
        task.verify_message = (
            f"Verified: {len(task.files_modified)} file(s) confirmed: {task.files_modified}"
        )

    if verbose:
        status = "✓" if task.verify_passed else "✗"
        print(f"  [{status}] {task.verify_message}")

    task.status = "done"
    return {"plan": plan, "file_index": file_index_updates}


# ── Summarizer ────────────────────────────────────────────────────────────────

SUMMARIZER_SYSTEM = """Summarize in ONE sentence what was accomplished.
Be specific: file name, what changed, outcome. No filler."""


def summarizer_node(state: ShukiState) -> dict:
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx:  int           = state["current_task_idx"]
    if idx >= len(plan):
        return {"current_task_idx": idx + 1}

    task = plan[idx]

    verify_msg   = getattr(task, "verify_message", "")
    executor_out = getattr(task, "executor_output", "")

    prompt = (
        f"Task: {task.description}\n"
        f"Outcome: {verify_msg or executor_out[:200] or 'completed'}"
    )

    session = BudgetedSession(
        system_prompt=SUMMARIZER_SYSTEM,
        tools=None,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )
    response = session.invoke(prompt)
    task.result_summary = str(response.content).strip()

    if verbose:
        print(f"[Summarizer] {task.result_summary}")

    return {"plan": plan, "current_task_idx": idx + 1}


# ── Finalizer ─────────────────────────────────────────────────────────────────

FINALIZER_SYSTEM = """Write a clear, concise response to the user's original request.
Based only on the completed subtask summaries provided.
Mention what was created, changed, or found. Be specific."""


def finalizer_node(state: ShukiState) -> dict:
    verbose = config.verbose
    plan = state.get("plan", [])
    summaries = "\n".join(
        f"- {t.title}: {t.result_summary or 'completed'}" for t in plan
    )
    session = BudgetedSession(
        system_prompt=FINALIZER_SYSTEM,
        tools=None,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )
    response = session.invoke(
        f"User request: {state['user_request']}\n\nCompleted:\n{summaries}"
    )
    answer = str(response.content)
    if verbose:
        print(f"\n[Finalizer]\n{answer}")
    return {"final_answer": answer}


# ── Routers ───────────────────────────────────────────────────────────────────

def route_start(state: ShukiState) -> str:
    """Decide whether to run discovery or jump straight to planning."""
    if _needs_discovery(state["user_request"]):
        return "discovery"
    return "planner"


def route_after_planner(state: ShukiState) -> str:
    """If the planner didn't produce a plan, it likely wants more discovery."""
    if not state.get("plan"):
        return "discovery"
    return "executor"


def route_after_verifier(state: ShukiState) -> str:
    """
    After verification:
    - If the write failed and we haven't retried yet → retry (back to executor)
    - Otherwise → summarize
    """
    plan = state.get("plan", [])
    idx  = state.get("current_task_idx", 0)
    if idx >= len(plan):
        return "summarize"

    task = plan[idx]
    verify_passed = getattr(task, "verify_passed", True)
    retry_count   = getattr(task, "retry_count", 0)

    if not verify_passed and retry_count < 1:
        task.retry_count = retry_count + 1
        if config.verbose:
            print(f"  [Router] Verification failed — retrying executor (attempt {task.retry_count})")
        return "retry"

    return "summarize"


def route_after_summarizer(state: ShukiState) -> str:
    plan = state.get("plan", [])
    idx  = state.get("current_task_idx", 0)
    if idx >= len(plan):
        return "finalize"
    current = plan[idx]
    for dep_id in current.depends_on:
        dep = next((t for t in plan if t.id == dep_id), None)
        if dep and dep.status != "done":
            return "finalize"
    return "continue"
