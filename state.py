"""
LangGraph state for the Shuki agent.

Each subtask carries its own selected rules, skills, and tools —
chosen by dedicated pipeline steps before the executor runs.
"""
from __future__ import annotations
from typing import Annotated, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import MessagesState


# ── Subtask plan entry ────────────────────────────────────────────────────────

@dataclass
class SubTask:
    id: int
    title: str                              # short label, e.g. "Read main.py"
    description: str                        # what the LLM needs to do
    depends_on: list[int]                   # ids of tasks that must finish first
    context_hints: list[str]                # filenames / keywords to inject

    # Set by pipeline steps (rules → skill → tool selector → executor)
    selected_rules: list[str] = field(default_factory=list)    # rule file contents
    selected_skills: list[str] = field(default_factory=list)   # skill names matched
    skill_prompt: str = ""                                      # merged skill system prompt
    selected_tool_names: list[str] = field(default_factory=list)  # tool names for executor

    # Legacy hint from planner (used by tool selector as a soft signal)
    tool_hint: Optional[str] = None

    # Re-planning depth guard — incremented each time this task triggers a re-plan
    replan_depth: int = 0

    # Filled after execution
    result_summary: Optional[str] = None
    tool_calls_made: list[dict] = field(default_factory=list)
    status: str = "pending"   # pending | rules | skills | tools | running | done | failed


# ── Agent state ───────────────────────────────────────────────────────────────

class ShukiState(MessagesState):
    # Original user request
    user_request: str

    # Plan produced by the planner (ordered list)
    plan: list[SubTask]

    # Index of the currently executing subtask
    current_task_idx: int

    # Workspace file index: filename -> last-known summary (from read/write ops)
    file_index: Annotated[dict[str, str], operator.or_]   # merge dicts on update

    # Accumulated raw tool outputs (kept small via summarisation)
    task_results: Annotated[list[dict], operator.add]

    # Final answer assembled after all tasks
    final_answer: Optional[str]

    # Error state
    error: Optional[str]


def initial_state(user_request: str) -> dict:
    return {
        "messages": [],
        "user_request": user_request,
        "plan": [],
        "current_task_idx": 0,
        "file_index": {},
        "task_results": [],
        "final_answer": None,
        "error": None,
    }
