"""
Rules Loader and Selector

Loads user rules from:
  1. %LOCALAPPDATA%/.shuki/rules/   (global user rules)
  2. {cwd}/.shuki/                  (project-local rules)

Both locations are merged. For each subtask, an LLM call selects only
the rules relevant to that specific task — so the executor context stays small
even if the user has many rules.

Rule files: any .md or .txt file in the rules directories.
Each file is one rule "document". Files can be named anything descriptive,
e.g. "no_print_statements.md", "always_use_type_hints.txt".
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

from config import config
from agent.llm_client import BudgetedSession


# ── Rule file discovery ───────────────────────────────────────────────────────

def _rules_dirs() -> list[Path]:
    """Return all directories that may contain rule files."""
    dirs = []

    # 1. Global user rules: %LOCALAPPDATA%\.shuki\rules  (or config override)
    from config import config as _cfg
    if _cfg.paths.global_shuki_dir:
        global_rules = Path(_cfg.paths.global_shuki_dir) / "rules"
    else:
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        global_rules = Path(local_app_data) / ".shuki" / "rules"
    if global_rules.exists():
        dirs.append(global_rules)

    # 2. Project-local rules: {workspace}/.shuki  (no subdir — flat)
    local_shuki = Path(config.workspace.root) / ".shuki"
    if local_shuki.exists():
        dirs.append(local_shuki)

    # 3. CWD .shuki (in case cwd != workspace root)
    cwd_shuki = Path.cwd() / ".shuki"
    if cwd_shuki.exists() and cwd_shuki != local_shuki:
        dirs.append(cwd_shuki)

    return dirs


def load_all_rules() -> dict[str, str]:
    """
    Return all available rules as {name: content}.

    Rules from later directories (project-local) shadow global ones
    with the same filename — allowing project overrides.
    """
    rules: dict[str, str] = {}
    for d in _rules_dirs():
        for fp in sorted(d.iterdir()):
            if fp.is_file() and fp.suffix in (".md", ".txt"):
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        rules[fp.stem] = content   # later dirs win on collision
                except OSError:
                    pass
    return rules


# ── Relevance selector ────────────────────────────────────────────────────────

_RULES_SELECTOR_SYSTEM = """You are a relevance filter for coding assistant rules.

Given a task description and a numbered list of rules, return ONLY the numbers
of rules that are directly relevant to completing that specific task.

Rules are relevant if they constrain HOW the task should be done (coding style,
forbidden patterns, required patterns, language preferences, etc.).

Respond with ONLY a comma-separated list of numbers, e.g.:  1,3,5
If no rules are relevant, respond with: none
No explanation, no other text."""


def select_relevant_rules(task_description: str, all_rules: dict[str, str]) -> list[str]:
    """
    Use a small LLM call to pick which rules apply to this specific task.
    Returns a list of rule content strings (not names).
    """
    if not all_rules:
        return []

    # Build a compact numbered index for the LLM
    names = list(all_rules.keys())
    # Show only names + first 80 chars of each rule (to stay small)
    index_lines = []
    for i, name in enumerate(names, 1):
        preview = all_rules[name][:80].replace("\n", " ")
        index_lines.append(f"{i}. [{name}] {preview}")
    index_str = "\n".join(index_lines)

    prompt = (
        f"Task: {task_description[:300]}\n\n"
        f"Rules:\n{index_str}"
    )

    session = BudgetedSession(
        system_prompt=_RULES_SELECTOR_SYSTEM,
        tools=None,
        max_tokens=100,
        verbose=config.verbose,
    )

    try:
        response = session.invoke(prompt)
        raw = str(response.content).strip().lower()

        if raw == "none" or not raw:
            return []

        # Parse comma-separated numbers
        selected_contents = []
        for token in re.split(r"[,\s]+", raw):
            token = token.strip()
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(names):
                    selected_contents.append(all_rules[names[idx]])

        if config.verbose:
            selected_names = [names[int(t)-1] for t in re.findall(r"\d+", raw)
                              if 0 < int(t) <= len(names)]
            print(f"  [Rules] Selected: {selected_names or 'none'}")

        return selected_contents

    except Exception as e:
        if config.verbose:
            print(f"  [Rules] Selector failed: {e}, using no rules")
        return []


def format_rules_for_context(rule_contents: list[str]) -> str:
    """Format selected rules into a block for injection into executor context."""
    if not rule_contents:
        return ""
    lines = ["=== RULES (must follow) ==="]
    for i, rule in enumerate(rule_contents, 1):
        # Truncate each rule to keep context small
        truncated = rule[:200]
        lines.append(f"{i}. {truncated}")
    return "\n".join(lines)
