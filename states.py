"""
LangGraph state for the Codex agent.

Designed to carry the minimum information needed across nodes
while supporting a small-context LLM executor.
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
        title: str                          # short label, e.g. "Read main.py"
            description: str                    # what the LLM needs to do
                depends_on: list[int]               # ids of tasks that must finish first
                    context_hints: list[str]            # filenames / keywords to inject
                        tool_hint: Optional[str] = None     # primary tool expected: read/write/run/search
                            result_summary: Optional[str] = None   # filled after execution
                                tool_calls_made: list[dict] = field(default_factory=list)
                                    status: str = "pending"             # pending | running | done | failed


                                    # ── Agent state ───────────────────────────────────────────────────────────────

                                    class CodexState(MessagesState):
                                        # Original user request
                                            user_request: str

                                                # Plan produced by the planner (ordered list)
                                                    plan: list[SubTask]

                                                        # Index of the currently executing subtask
                                                            current_task_idx: int

                                                                # Workspace file index: filename -> last-known summary (from read/write ops)
                                                                    file_index: Annotated[dict[str, str], operator.or_]   # merge dicts

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