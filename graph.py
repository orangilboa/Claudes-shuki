"""
LangGraph graph for the Shuki agent.

Full pipeline per subtask:
  planner
    └─► skill_picker ──[multi-skill]──► replanner ──► skill_picker (loop)
          └─[single/no skill]─► rules_selector
                                      └─► tool_selector
                                                └─► executor
                                                      └─► summarizer ──[more tasks]──► skill_picker
                                                                        [done]──────► finalizer
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ShukiState
from agent.nodes import (
    planner_node,
    rules_selector_node,
    skill_picker_node,
    replanner_node,
    tool_selector_node,
    executor_node,
    summarizer_node,
    finalizer_node,
    route_after_skill_picker,
    route_after_summarizer,
)


def build_graph(checkpointer=None):
    """
    Build and compile the Shuki LangGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for resumable runs.
    """
    builder = StateGraph(ShukiState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("planner",        planner_node)
    builder.add_node("rules_selector", rules_selector_node)
    builder.add_node("skill_picker",   skill_picker_node)
    builder.add_node("replanner",      replanner_node)
    builder.add_node("tool_selector",  tool_selector_node)
    builder.add_node("executor",       executor_node)
    builder.add_node("summarizer",     summarizer_node)
    builder.add_node("finalizer",      finalizer_node)

    # ── Edges ──────────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "skill_picker")

    # Skills → re-plan OR rules selection
    builder.add_conditional_edges(
        "skill_picker",
        route_after_skill_picker,
        {
            "replan":       "replanner",
            "select_rules": "rules_selector",
        },
    )

    # Re-planner → back to skills (new subtasks restart the pipeline)
    builder.add_edge("replanner", "skill_picker")

    # Rules → tool selection
    builder.add_edge("rules_selector", "tool_selector")
    builder.add_edge("executor",      "summarizer")

    # Summarizer → next task (loop) or done
    builder.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "continue":  "skill_picker",   # next subtask enters pipeline from top
            "finalize":  "finalizer",
        },
    )

    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)


def build_graph_with_memory():
    """Convenience: build with in-memory checkpointer."""
    return build_graph(checkpointer=MemorySaver())
