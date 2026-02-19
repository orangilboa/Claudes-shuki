"""
LangGraph graph for the Shuki agent.

Per-subtask pipeline:
  planner
    └─► skill_picker ──[multi-skill]──► replanner ──┐
          └─► rules_selector                        │ (loops back)
                └─► tool_selector                  │
                      └─► reasoner  (read tools only, outputs edit plan)
                            └─► writer    (no LLM, applies plan mechanically)
                                  └─► verifier ──[fail+retry]──► reasoner
                                        └─► summarizer ──[more]──► skill_picker
                                                          [done]──► finalizer
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import ShukiState
from agent.nodes import (
    discovery_node,
    planner_node,
    skill_picker_node,
    replanner_node,
    rules_selector_node,
    tool_selector_node,
    reasoner_node,
    writer_node,
    verifier_node,
    summarizer_node,
    finalizer_node,
    route_start,
    route_after_planner,
    route_after_skill_picker,
    route_after_verifier,
    route_after_summarizer,
)


def build_graph(checkpointer=None):
    builder = StateGraph(ShukiState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("discovery",      discovery_node)
    builder.add_node("planner",        planner_node)
    builder.add_node("skill_picker",   skill_picker_node)
    builder.add_node("replanner",      replanner_node)
    builder.add_node("rules_selector", rules_selector_node)
    builder.add_node("tool_selector",  tool_selector_node)
    builder.add_node("reasoner",       reasoner_node)
    builder.add_node("writer",         writer_node)
    builder.add_node("verifier",       verifier_node)
    builder.add_node("summarizer",     summarizer_node)
    builder.add_node("finalizer",      finalizer_node)

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
            "discovery":    "discovery",
            "skill_picker": "skill_picker",
        }
    )

    builder.add_conditional_edges(
        "skill_picker",
        route_after_skill_picker,
        {
            "replan":       "replanner",
            "select_rules": "rules_selector",
        },
    )

    builder.add_edge("replanner",      "skill_picker")   # new tasks re-enter from top
    builder.add_edge("rules_selector", "tool_selector")
    builder.add_edge("tool_selector",  "reasoner")
    builder.add_edge("reasoner",       "writer")
    builder.add_edge("writer",         "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "retry":     "reasoner",    # one retry with real file content in state
            "summarize": "summarizer",
        },
    )

    builder.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "continue": "skill_picker",  # next subtask enters pipeline from top
            "finalize": "finalizer",
        },
    )

    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)


def build_graph_with_memory():
    return build_graph(checkpointer=MemorySaver())
