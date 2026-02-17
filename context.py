"""
Context Assembler

Builds the smallest possible context to inject into the executor LLM
for a given subtask. Respects the token budget strictly.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from config import config
from agent.state import SubTask, CodexState


class ContextAssembler:
    """
    Assembles the minimal context string for a subtask executor call.

    Budget allocation (chars):
      - Subtask description   : always included
      - Prior summaries       : config.llm.summary_max_chars each, most-recent first
      - File snippets         : config.llm.file_snippet_max_chars each
      - Total cap             : executor_budget_tokens * chars_per_token
    """

    def __init__(self):
        self.ws = Path(config.workspace.root).resolve()
        self.budget = int(
            config.llm.executor_budget_tokens * config.llm.chars_per_token
        )

    def build(self, task: SubTask, state: CodexState) -> str:
        """Return a context string ready to prepend to the executor system prompt."""
        parts: list[str] = []
        remaining = self.budget

        # 1. Task description (always)
        task_block = f"=== CURRENT TASK: {task.title} ===\n{task.description}"
        parts.append(task_block)
        remaining -= len(task_block)

        # 2. Summaries of dependency tasks (in dependency order)
        dep_summaries = self._collect_dep_summaries(task, state)
        if dep_summaries and remaining > 100:
            block = "=== PRIOR TASK RESULTS ===\n" + dep_summaries
            block = block[:remaining - 50]
            parts.append(block)
            remaining -= len(block)

        # 3. Relevant file snippets hinted by the planner
        for hint in task.context_hints:
            if remaining < 150:
                break
            snippet = self._fetch_snippet(hint, state)
            if snippet:
                snippet = snippet[:config.llm.file_snippet_max_chars]
                parts.append(snippet)
                remaining -= len(snippet)

        return "\n\n".join(parts)

    # ── internals ─────────────────────────────────────────────────────────────

    def _collect_dep_summaries(self, task: SubTask, state: CodexState) -> str:
        """Collect summaries of completed dependency tasks."""
        lines = []
        plan = state.get("plan", [])
        for dep_id in task.depends_on:
            dep = next((t for t in plan if t.id == dep_id), None)
            if dep and dep.result_summary:
                summary = dep.result_summary[:config.llm.summary_max_chars]
                lines.append(f"[Task {dep_id} - {dep.title}]: {summary}")
        return "\n".join(lines)

    def _fetch_snippet(self, hint: str, state: CodexState) -> Optional[str]:
        """
        Try to get a small snippet for a context hint.
        Hint can be a filename or a keyword.
        """
        # Check file index first (already-read files)
        file_index: dict = state.get("file_index", {})
        if hint in file_index:
            return f"[Cached info for {hint}]:\n{file_index[hint][:config.llm.file_snippet_max_chars]}"

        # Try as a file path
        candidate = self.ws / hint
        if candidate.exists() and candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
                # Only grab first N chars (head of file for context)
                snippet = content[:config.llm.file_snippet_max_chars]
                return f"[File snippet: {hint}]\n{snippet}"
            except Exception:
                pass

        # Try fuzzy match against workspace files
        for fp in self.ws.rglob("*"):
            if fp.is_file() and hint.lower() in fp.name.lower():
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    snippet = content[:config.llm.file_snippet_max_chars]
                    rel = fp.relative_to(self.ws)
                    return f"[File snippet: {rel}]\n{snippet}"
                except Exception:
                    pass

        return None
