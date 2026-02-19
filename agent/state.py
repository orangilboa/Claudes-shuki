"""
LangGraph state for the Shuki agent.

The planner assigns a skill and tool list to each subtask.
The executor runs directly against those assignments.
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

    # Set by planner
    skill: str = "generic"                  # skill name matched to a .shuki/skill/*.md stem
    tools: list[str] = field(default_factory=list)  # tool names assigned by planner

    # Set by executor
    executor_output: str = ""               # raw final response from executor
    files_modified: list[str] = field(default_factory=list)  # for verifier

    # Set by verifier
    verify_passed: bool = True
    verify_message: str = ""

    # Retry counter — verifier allows one executor retry on failure
    retry_count: int = 0

    # Filled after execution
    result_summary: Optional[str] = None
    tool_calls_made: list[dict] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | failed


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

    # Results from the discovery step (file lists, search matches, etc.)
    discovery_results: str

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
        "discovery_results": "",
        "final_answer": None,
        "error": None,
    }
