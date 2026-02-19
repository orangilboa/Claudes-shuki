"""
LangGraph graph for the Shuki agent.

Pipeline:
  START → [route_start] → discovery | planner
  discovery → planner
  planner → [route_after_planner] → executor | discovery
  executor → verifier
  verifier → [route_after_verifier] → executor (retry) | summarizer
  summarizer → [route_after_summarizer] → executor (next task) | finalizer
  finalizer → END
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ShukiState
from agent.nodes import (
    discovery_node,
    planner_node,
    executor_node,
    verifier_node,
    summarizer_node,
    finalizer_node,
    route_start,
    route_after_planner,
    route_after_verifier,
    route_after_summarizer,
)


def build_graph(checkpointer=None):
    builder = StateGraph(ShukiState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("discovery",  discovery_node)
    builder.add_node("planner",    planner_node)
    builder.add_node("executor",   executor_node)
    builder.add_node("verifier",   verifier_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("finalizer",  finalizer_node)

    # ── Edges ──────────────────────────────────────────────────────────────────

    builder.add_conditional_edges(
        START,
        route_start,
        {
            "discovery": "discovery",
            "planner":   "planner",
        }
    )

    builder.add_edge("discovery", "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "discovery": "discovery",
            "executor":  "executor",
        }
    )

    builder.add_edge("executor", "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "retry":     "executor",    # one retry with error context
            "summarize": "summarizer",
        },
    )

    builder.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "continue": "executor",   # next subtask enters pipeline
            "finalize": "finalizer",
        },
    )

    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)


def build_graph_with_memory():
    return build_graph(checkpointer=MemorySaver())
