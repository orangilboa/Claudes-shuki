"""
LangGraph graph for the Codex agent.

Flow:
  START → planner → executor → summarizer → [loop or finalize] → finalizer → END

  The executor+summarizer pair loops until all subtasks are done.
  """
  from __future__ import annotations
  from langgraph.graph import StateGraph, START, END
  from langgraph.checkpoint.memory import MemorySaver

  from agent.state import CodexState
  from agent.nodes import (
      planner_node,
          executor_node,
              summarizer_node,
                  finalizer_node,
                      should_continue,
                      )


                      def build_graph(checkpointer=None):
                          """
                              Build and compile the Codex LangGraph.

                                  Args:
                                          checkpointer: Optional LangGraph checkpointer for resumable runs.
                                                                Pass MemorySaver() for in-memory, or SqliteSaver for persistence.
                                                                    """
                                                                        builder = StateGraph(CodexState)

                                                                            # ── Add nodes ──────────────────────────────────────────────────────────────
                                                                                builder.add_node("planner", planner_node)
                                                                                    builder.add_node("executor", executor_node)
                                                                                        builder.add_node("summarizer", summarizer_node)
                                                                                            builder.add_node("finalizer", finalizer_node)

                                                                                                # ── Edges ──────────────────────────────────────────────────────────────────
                                                                                                    # Always start with planning
                                                                                                        builder.add_edge(START, "planner")
                                                                                                            builder.add_edge("planner", "executor")

                                                                                                                # After summarizing, route: keep looping or finalize
                                                                                                                    builder.add_conditional_edges(
                                                                                                                            "summarizer",
                                                                                                                                    should_continue,
                                                                                                                                            {
                                                                                                                                                        "execute": "executor",
                                                                                                                                                                    "finalize": "finalizer",
                                                                                                                                                                            },
                                                                                                                                                                                )

                                                                                                                                                                                    # Executor always feeds into summarizer
                                                                                                                                                                                        builder.add_edge("executor", "summarizer")

                                                                                                                                                                                            # Done
                                                                                                                                                                                                builder.add_edge("finalizer", END)

                                                                                                                                                                                                    # ── Compile ────────────────────────────────────────────────────────────────
                                                                                                                                                                                                        return builder.compile(checkpointer=checkpointer)


                                                                                                                                                                                                        def build_graph_with_memory():
                                                                                                                                                                                                            """Convenience: build graph with in-memory checkpointer."""
                                                                                                                                                                                                                return build_graph(checkpointer=MemorySaver())