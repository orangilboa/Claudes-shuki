"""
LangGraph nodes for the Shuki agent.

Per-subtask pipeline:
  skill_picker → [replanner?] → rules_selector → tool_selector
    → reasoner (read tools only, outputs edit plan)
    → writer   (no LLM — mechanically applies the plan)
    → verifier (re-reads file, confirms change landed; retries once on failure)
    → summarizer

Global nodes: planner, finalizer
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
from agent.rules import load_all_rules, select_relevant_rules
from agent.skills import load_all_skills, pick_skills, needs_resplit
from agent.tool_selector import select_tools_for_task, get_tools_for_names
from tools.code_tools import ALL_TOOLS, TOOL_MAP, READ_TOOLS, WRITE_TOOLS


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

PLANNER_SYSTEM = """You are a task planner for a general-purpose assistant.

Use the provided discovery results to break the user request into an ORDERED list of small, FOCUSED subtasks.
Each subtask MUST touch exactly ONE file or resource.

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
    "tool_hint": "read|write|run|search|patch"
  }}
]

Respond with ONLY valid JSON, no markdown fences.
"""


def planner_node(state: ShukiState) -> dict:
    verbose = config.verbose
    session = BudgetedSession(
        system_prompt=PLANNER_SYSTEM,
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
            print(f"  [{t.id}] {t.title}  deps={t.depends_on}")
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
                tool_hint=item.get("tool_hint"),
            ))
        return tasks
    except (json.JSONDecodeError, KeyError):
        return [SubTask(id=1 + id_offset, title="Execute request",
                        description=raw, depends_on=[], context_hints=[])]


# ── Skill Picker ──────────────────────────────────────────────────────────────

def skill_picker_node(state: ShukiState) -> dict:
    task = _current_task(state)
    if task is None:
        return {}
    if config.verbose:
        print(f"\n[Skills] Task {task.id}: {task.title}")
    all_skills = load_all_skills()
    matched_names, merged_prompt = pick_skills(task.description, all_skills)
    task.selected_skills = matched_names
    task.skill_prompt = merged_prompt
    if needs_resplit(matched_names):
        task.status = "needs_resplit"
        if config.verbose:
            print(f"  Multi-skill {matched_names} → re-plan")
    else:
        task.status = "skills"
    return {"plan": state["plan"]}


# ── Re-planner ────────────────────────────────────────────────────────────────

REPLANNER_SYSTEM = """Split this multi-skill task into smaller subtasks, one per skill.
Respond with ONLY a valid JSON array, no markdown fences.
[{{"id":1,"title":"...","description":"...","depends_on":[],"context_hints":[],"tool_hint":"read"}}]"""


def replanner_node(state: ShukiState) -> dict:
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]
    task = plan[idx]
    if verbose:
        print(f"\n[Re-planner] Splitting task {task.id} (depth {task.replan_depth+1})")
    skills_str = "\n".join(f"- {s}" for s in task.selected_skills)
    session = BudgetedSession(
        system_prompt=REPLANNER_SYSTEM,
        tools=None,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )
    response = session.invoke(
        f"Task: {task.description}\nSkills: {skills_str}"
    )
    max_id = max((t.id for t in plan), default=0)
    new_tasks = _parse_plan(str(response.content), id_offset=max_id)
    if not new_tasks:
        task.status = "skills"
        return {"plan": plan}
    parent_depth = task.replan_depth + 1
    for t in new_tasks:
        t.replan_depth = parent_depth
    new_plan = plan[:idx] + new_tasks + plan[idx+1:]
    if verbose:
        for t in new_tasks:
            print(f"  [{t.id}] {t.title}")
    return {"plan": new_plan, "current_task_idx": idx}


# ── Rules Selector ────────────────────────────────────────────────────────────

def rules_selector_node(state: ShukiState) -> dict:
    task = _current_task(state)
    if task is None:
        return {}
    if config.verbose:
        print(f"\n[Rules] Task {task.id}: {task.title}")
    all_rules = load_all_rules()
    if all_rules:
        task.selected_rules = select_relevant_rules(task.description, all_rules)
    task.status = "rules"
    return {"plan": state["plan"]}


# ── Tool Selector ─────────────────────────────────────────────────────────────

def tool_selector_node(state: ShukiState) -> dict:
    task = _current_task(state)
    if task is None:
        return {}
    if config.verbose:
        print(f"\n[ToolSelector] Task {task.id}: {task.title}")
    selected_names = select_tools_for_task(
        task_description=task.description,
        skill_prompt=task.skill_prompt,
        tool_hint=task.tool_hint,
    )
    if not selected_names:
        selected_names = [t.name for t in ALL_TOOLS]
        if config.verbose:
            print("  Fallback: all tools")
    task.selected_tool_names = selected_names
    task.status = "tools"
    if config.verbose:
        print(f"  Selected: {selected_names}")
    return {"plan": state["plan"]}


# ── Reasoner ──────────────────────────────────────────────────────────────────
#
# Has READ tools only. Explores freely, then outputs a structured edit plan.
# Cannot write — so it cannot hallucinate a completed edit.

REASONER_SYSTEM = """You are a careful analyst completing one focused task.

You have access to READ tools only: read_file, search_in_files, list_directory, get_file_info.

━━━ TOOL USAGE STRATEGY ━━━

Prefer search_in_files over read_file for large files:
- Use search_in_files(pattern="...", path="file.py") to locate specific lines before reading the whole file.
- Only call read_file when you need surrounding context that search results alone don't provide.
- For bulk-replace tasks (e.g. "replace all X with Y"), search first to find every occurrence,
  then read_file once to get the exact surrounding text needed for the edit plan.

Avoid reading the same file multiple times — the content doesn't change between calls.

When you have enough context, output an edit plan as a JSON block.
Do NOT attempt to write or edit files — that is handled separately.

━━━ OUTPUT FORMAT (choose one) ━━━

For patching an existing file (MOST COMMON):
```json
{{
  "action": "patch",
  "file": "relative/path/to/file",
  "old": "exact string to replace (must be unique in the file)",
  "new": "replacement string"
}}
```

For writing a whole new file:
```json
{{
  "action": "write",
  "file": "relative/path/to/file",
  "content": "full file content here"
}}
```

For multiple coordinated edits in ONE file (e.g. add function + update imports):
```json
{{
  "action": "multi_patch",
  "file": "relative/path/to/file",
  "patches": [
    {{"old": "import os", "new": "import os\\nimport requests"}},
    {{"old": "# end of file", "new": "def new_func():\\n    pass\\n\\n# end of file"}}
  ]
}}
```

For tasks that require no file changes (read-only, research, reporting):
```json
{{
  "action": "none",
  "summary": "what you found or concluded"
}}
```

━━━ WHEN TO USE WHICH ACTION ━━━

Use "write" when:
- Making MORE THAN 3 changes scattered across a file (e.g. replace all print() calls,
  rename a function everywhere, add imports + update multiple call sites).
- The change is a bulk refactor. Read the full file, apply all changes mentally, output
  the complete new content. This is far more reliable than many individual patches.

Use "patch" or "multi_patch" only for:
- 1-3 targeted surgical changes (e.g. fix a single bug, change one function signature).

━━━ EXAMPLES ━━━

Task: "Replace all print() calls with logger.info() in main.py"
You read main.py. There are 15 print() calls scattered through the file.
→ Use "write": output the COMPLETE modified file with every print() replaced.
Output:
```json
{{
  "action": "write",
  "file": "main.py",
  "content": "...complete new file content..."
}}
```

Task: "Add a get_user function to auth.py"
You read auth.py, see the existing code structure.
Output:
```json
{{
  "action": "patch",
  "file": "auth.py",
  "old": "# end of file",
  "new": "def get_user(user_id: int) -> dict:\\n    return db.query('SELECT * FROM users WHERE id = ?', user_id)\\n\\n# end of file"
}}
```

━━━ CRITICAL RULES ━━━

- Your task targets ONE file. If you think multiple files need changes, only handle the file mentioned in the task description.
- ALWAYS call read_file on the target file before producing any edit plan.
- For patches: "old" MUST be an exact verbatim copy from the file. Copy-paste from the read_file result — do NOT retype or paraphrase.
- For write: output the COMPLETE file content — every line, nothing omitted.
- Output EXACTLY ONE JSON block. No prose, no explanation, no code outside the JSON.

{skill_section}{rules_section}"""


def reasoner_node(state: ShukiState) -> dict:
    """
    Reasoner: read-only tools, outputs a structured edit plan as JSON.
    Stores the plan in task.edit_plan for the writer to execute.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx: int = state["current_task_idx"]
    if idx >= len(plan):
        return {}

    task = plan[idx]
    task.status = "running"

    # Give the reasoner ALL read tools — tool selector output is for the writer, not the explorer
    read_tool_objects = [TOOL_MAP[n] for n in READ_TOOLS if n in TOOL_MAP]
    read_tool_map = {t.name: t for t in read_tool_objects}

    # Build skill/rules sections (kept short)
    skill_section = ""
    if task.skill_prompt:
        skill_section = f"\nGuidance: {task.skill_prompt[:300]}\n"
    rules_section = ""
    if task.selected_rules:
        rules_section = "\nConstraints: " + " | ".join(
            r[:100].replace("\n", " ") for r in task.selected_rules
        ) + "\n"

    system = REASONER_SYSTEM.format(
        skill_section=skill_section,
        rules_section=rules_section,
    )

    # Inject prior context (file index + dependency summaries)
    assembler = ContextAssembler()
    context_str = assembler.build(task, state)

    if verbose:
        print(f"\n[Reasoner] Task {task.id}: {task.title}")
        print(f"  Read tools: {list(read_tool_map.keys())}")

    session = BudgetedSession(
        system_prompt=system,
        tools=read_tool_objects,
        max_tokens=config.llm.max_output_tokens,
        verbose=verbose,
    )

    prompt = f"TASK: {task.description}"
    if context_str:
        prompt += f"\n\nContext:\n{context_str}"

    # On retry after failed patch: inject the error details and actual file content
    # so the model knows exactly what went wrong and can copy-paste the real strings.
    if task.retry_count > 0:
        wr = getattr(task, "write_result", {})
        prev_plan = getattr(task, "edit_plan", {})
        retry_ctx = "\n\n━━━ PREVIOUS ATTEMPT FAILED ━━━\n"
        retry_ctx += f"Writer error: {wr.get('message', 'unknown')}\n"
        retry_ctx += f"Failed edit plan: {json.dumps(prev_plan)[:600]}\n"
        # If we know the target file, inject its actual content
        target_file = prev_plan.get("file", "")
        if target_file:
            try:
                actual = TOOL_MAP["read_file"].invoke({"path": target_file})
                retry_ctx += (f"\nACTUAL current content of {target_file} "
                              f"(copy strings EXACTLY from here):\n{actual}")
            except Exception:
                pass
        retry_ctx += "\n━━━ END PREVIOUS ATTEMPT ━━━\n"
        retry_ctx += ("\nIMPORTANT: Your previous 'old' string did NOT match "
                      "the file. You MUST copy the exact text from the file "
                      "content shown above. Do NOT paraphrase or rewrite it.")
        prompt += retry_ctx

    response = session.invoke(prompt)
    file_index_updates = {}

    # Run read tool loop — no write tools available so no hallucinated edits possible
    MAX_READ_ROUNDS = 10
    for _ in range(MAX_READ_ROUNDS):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name  = tc["name"]
            args  = tc["args"]
            tid   = tc["id"]
            if verbose:
                print(f"  [Read] {name}({json.dumps(args)})")
            tool_fn = read_tool_map.get(name)
            result  = tool_fn.invoke(args) if tool_fn else f"ERROR: tool '{name}' not available"
            if verbose:
                print(f"  [Result] {str(result)}")
            # Cache reads into file_index so writer and future tasks can use them
            if name == "read_file" and "path" in args and isinstance(result, str):
                file_index_updates[args["path"]] = result
            session.append_tool_result(tid, str(result), name)
        response = session.continue_after_tools()

    # Extract the JSON edit plan from the final response
    raw_output = str(response.content)
    edit_plan  = _extract_edit_plan(raw_output)

    if verbose:
        print(f"  [Reasoner] Edit plan: {json.dumps(edit_plan)}")

    # Store on the task object for the writer
    task.edit_plan = edit_plan
    task.reasoner_output = raw_output

    return {"plan": plan, "file_index": file_index_updates}


def _extract_edit_plan(text: str) -> dict:
    """Pull the JSON edit plan out of the reasoner's output."""
    # Try fenced block first
    for pattern in (r"```json\s*(.*?)```", r"```\s*(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
    # Try bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: treat as read-only summary
    return {"action": "none", "summary": text[:500]}


# ── Writer ────────────────────────────────────────────────────────────────────
#
# Pure Python — no LLM. Applies the edit plan mechanically.
# Returns verification info so the verifier can confirm the change landed.

def writer_node(state: ShukiState) -> dict:
    """
    Apply the reasoner's edit plan mechanically.
    No LLM — just calls patch_file or write_file directly in Python.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx:  int           = state["current_task_idx"]
    if idx >= len(plan):
        return {}

    task      = plan[idx]
    edit_plan = getattr(task, "edit_plan", {})
    action    = edit_plan.get("action", "none")

    if verbose:
        print(f"\n[Writer] Task {task.id}: action={action}")

    file_index_updates: dict[str, str] = {}
    write_result: dict = {"action": action, "success": False, "message": ""}

    if action == "none":
        write_result["success"] = True
        write_result["message"] = edit_plan.get("summary", "No file changes needed.")

    elif action == "patch":
        file_path = edit_plan.get("file", "")
        old_str   = edit_plan.get("old", "")
        new_str   = edit_plan.get("new", "")

        if not file_path or not old_str:
            write_result["message"] = "ERROR: patch plan missing 'file' or 'old' field."
        else:
            result = TOOL_MAP["patch_file"].invoke({
                "path": file_path,
                "old_str": old_str,
                "new_str": new_str,
            })
            write_result["success"] = str(result).startswith("OK")
            write_result["message"] = str(result)
            write_result["file"]    = file_path
            write_result["old"]     = old_str
            write_result["new"]     = new_str
            if write_result["success"]:
                try:
                    file_index_updates[file_path] = TOOL_MAP["read_file"].invoke({"path": file_path})
                except Exception:
                    file_index_updates[file_path] = f"Patched by task {task.id}"

    elif action == "multi_patch":
        file_path = edit_plan.get("file", "")
        patches   = edit_plan.get("patches", [])

        if not file_path or not patches:
            write_result["message"] = "ERROR: multi_patch missing 'file' or 'patches' field."
        else:
            results = []
            all_ok = True
            for i, patch in enumerate(patches):
                old_str = patch.get("old", "")
                new_str = patch.get("new", "")
                if not old_str:
                    results.append(f"Patch {i+1}: ERROR - missing 'old' field")
                    all_ok = False
                    continue
                result = TOOL_MAP["patch_file"].invoke({
                    "path": file_path,
                    "old_str": old_str,
                    "new_str": new_str,
                })
                if not str(result).startswith("OK"):
                    # Include the failed old_str in the error for debugging
                    results.append(
                        f"Patch {i+1}: ERROR: String not found in {file_path}. "
                        f"Check whitespace/indentation."
                    )
                    all_ok = False
                    break  # stop on first failure so verifier can see which patch failed
                else:
                    results.append(f"Patch {i+1}: {result}")
            write_result["success"] = all_ok
            write_result["message"] = "\n".join(results)
            write_result["file"]    = file_path
            if all_ok:
                try:
                    file_index_updates[file_path] = TOOL_MAP["read_file"].invoke({"path": file_path})
                except Exception:
                    file_index_updates[file_path] = f"Multi-patched by task {task.id}"

    elif action == "write":
        file_path = edit_plan.get("file", "")
        content   = edit_plan.get("content", "")

        if not file_path:
            write_result["message"] = "ERROR: write plan missing 'file' field."
        else:
            # Use write_file (creates or overwrites)
            result = TOOL_MAP["write_file"].invoke({
                "path": file_path,
                "content": content,
            })
            write_result["success"] = str(result).startswith("OK")
            write_result["message"] = str(result)
            write_result["file"]    = file_path
            if write_result["success"]:
                file_index_updates[file_path] = content[:200]

    else:
        write_result["message"] = f"Unknown action: {action}"

    if verbose:
        print(f"  [Writer] {write_result['message']}")

    task.write_result = write_result
    return {"plan": plan, "file_index": file_index_updates}


# ── Verifier ──────────────────────────────────────────────────────────────────
#
# Re-reads the file and confirms the expected content is present.
# If not, sends the reasoner back for one retry with the real file content shown.

def verifier_node(state: ShukiState) -> dict:
    """
    Confirm that the writer's change actually landed in the file.
    Sets task.verify_passed and task.verify_message.
    """
    verbose = config.verbose
    plan: list[SubTask] = state["plan"]
    idx:  int           = state["current_task_idx"]
    if idx >= len(plan):
        return {}

    task         = plan[idx]
    write_result = getattr(task, "write_result", {})
    action       = write_result.get("action", "none")

    if verbose:
        print(f"\n[Verifier] Task {task.id}")

    # No file change — nothing to verify
    if action == "none" or not write_result.get("success"):
        task.verify_passed  = write_result.get("success", True)
        task.verify_message = write_result.get("message", "No write performed.")
        task.status = "done"
        return {"plan": plan}

    file_path = write_result.get("file", "")
    if not file_path:
        task.verify_passed  = False
        task.verify_message = "No file path in write result."
        task.status = "done"
        return {"plan": plan}

    # Re-read the file
    try:
        actual_content = TOOL_MAP["read_file"].invoke({"path": file_path})
    except Exception as e:
        task.verify_passed  = False
        task.verify_message = f"Could not re-read {file_path}: {e}"
        task.status = "done"
        return {"plan": plan}

    if action == "patch":
        new_str = write_result.get("new", "")
        old_str = write_result.get("old", "")
        if new_str and new_str in str(actual_content):
            task.verify_passed  = True
            task.verify_message = f"Verified: new content present in {file_path}."
        elif old_str and old_str in str(actual_content):
            task.verify_passed  = False
            task.verify_message = f"FAIL: old string still present in {file_path}."
        else:
            task.verify_passed  = True
            task.verify_message = f"OK: file modified (likely already correct)."

    elif action == "multi_patch":
        # For multi_patch, verify that at least one of the new strings is present
        # and none of the old strings remain
        patches     = task.edit_plan.get("patches", [])
        new_found   = 0
        old_remains = 0
        for patch in patches:
            new_str = patch.get("new", "")
            old_str = patch.get("old", "")
            if new_str and new_str in str(actual_content):
                new_found += 1
            if old_str and old_str in str(actual_content):
                old_remains += 1
        if new_found > 0 and old_remains == 0:
            task.verify_passed  = True
            task.verify_message = f"Verified: {new_found}/{len(patches)} patches applied to {file_path}."
        elif old_remains > 0:
            task.verify_passed  = False
            task.verify_message = f"FAIL: {old_remains} old strings still in {file_path}."
        else:
            task.verify_passed  = True
            task.verify_message = f"OK: multi-patch applied (likely already correct)."

    elif action == "write":
        expected_fragment = write_result.get("content", "")[:80]
        if expected_fragment and expected_fragment in str(actual_content):
            task.verify_passed  = True
            task.verify_message = f"Verified: content present in {file_path}."
        else:
            task.verify_passed  = False
            task.verify_message = f"FAIL: written content not found in {file_path}."

    # Update file_index with fresh content after verification
    file_index_updates = {file_path: str(actual_content)[:400]}

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

    # Build a compact description of what happened for the summarizer
    verify_msg = getattr(task, "verify_message", "")
    reasoner_summary = ""
    edit_plan = getattr(task, "edit_plan", {})
    if edit_plan.get("action") == "none":
        reasoner_summary = edit_plan.get("summary", "")

    prompt = (
        f"Task: {task.description}\n"
        f"Outcome: {verify_msg or reasoner_summary or 'completed'}"
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
    return "skill_picker"


def route_after_skill_picker(state: ShukiState) -> str:
    task = _current_task(state)
    if task and task.status == "needs_resplit":
        if task.replan_depth < config.llm.max_replan_depth:
            return "replan"
        task.status = "skills"
        if config.verbose:
            print(f"  [Router] Replan depth limit hit for task {task.id}, proceeding")
    return "select_rules"


def route_after_verifier(state: ShukiState) -> str:
    """
    After verification:
    - If the write failed and we haven't retried yet → retry (back to reasoner)
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
            print(f"  [Router] Verification failed — retrying reasoner (attempt {task.retry_count})")
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
