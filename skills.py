"""
Skill Loader and Picker

Skills are task-type definitions — markdown/txt files that describe HOW
to approach a category of work (coding, documentation, testing, git, etc.).

Discovery order (later paths win on name collision):
  1. {install_dir}/skills/          — bundled skills shipped with Shuki
  2. %LOCALAPPDATA%/.shuki/skills/  — user-defined extensions

Each skill file is a markdown document with:
  - A short description (first non-empty line or a # header)
  - Detailed instructions for the executor
  - Optional: a list of preferred tools

The skill picker:
  1. Builds a compact index (name + one-line description)
  2. LLM selects matching skill names
  3. If 0 skills match  → use generic fallback
     If 1 skill matches → inject its full prompt, proceed
     If 2+ skills match → signal for re-planning (caller handles split)
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

from config import config
from agent.llm_client import BudgetedSession


# ── Skill discovery ───────────────────────────────────────────────────────────

def _skill_dirs() -> list[Path]:
    """Return ordered list of skill directories."""
    dirs = []

    # 1. Bundled skills (relative to this file's package root, or config override)
    from config import config as _cfg
    if _cfg.paths.install_skills_dir:
        install_skills = Path(_cfg.paths.install_skills_dir)
    else:
        install_skills = Path(__file__).parent.parent / "skills"
    if install_skills.exists():
        dirs.append(install_skills)

    # 2. User-extended skills: %LOCALAPPDATA%\.shuki\skills
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    user_skills = Path(local_app_data) / ".shuki" / "skills"
    if user_skills.exists():
        dirs.append(user_skills)

    return dirs


def load_all_skills() -> dict[str, dict]:
    """
    Return all skills as {name: {"description": str, "content": str, "path": Path}}.
    User skills shadow bundled skills with the same filename stem.
    """
    skills: dict[str, dict] = {}
    for d in _skill_dirs():
        for fp in sorted(d.iterdir()):
            if fp.is_file() and fp.suffix in (".md", ".txt"):
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace").strip()
                    if not content:
                        continue
                    description = _extract_description(content)
                    skills[fp.stem] = {
                        "description": description,
                        "content": content,
                        "path": fp,
                    }
                except OSError:
                    pass
    return skills


def _extract_description(content: str) -> str:
    """Extract a one-line description from a skill file."""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip markdown heading markers
        line = re.sub(r"^#+\s*", "", line)
        return line[:120]
    return "(no description)"


# ── Skill picker ──────────────────────────────────────────────────────────────

_SKILL_PICKER_SYSTEM = """You are a task classifier for a coding assistant.

Given a task description and a numbered list of skill types, return the numbers
of ALL skills that apply to this task.

Respond with ONLY a comma-separated list of numbers, e.g.:  2,4
If only one skill applies, return just that number, e.g.: 1
If no skill fits, respond with: none
No explanation, no other text."""

# Fallback skill injected when no specific skill matches
_GENERIC_SKILL = """You are a precise, careful assistant. Complete the task exactly as described.
Think step by step. Prefer targeted edits over large rewrites. Verify your work."""


def pick_skills(
    task_description: str,
    all_skills: dict[str, dict],
) -> tuple[list[str], str]:
    """
    Select skills for a task.

    Returns:
        (matched_names, merged_prompt)

    matched_names: list of skill names that matched (empty = generic fallback used)
    merged_prompt: the skill system prompt to inject into the executor
    """
    if not all_skills:
        return [], _GENERIC_SKILL

    # Build compact index: number, name, one-line description
    names = list(all_skills.keys())
    index_lines = [f"{i}. [{name}] {all_skills[name]['description']}"
                   for i, name in enumerate(names, 1)]
    index_str = "\n".join(index_lines)

    prompt = (
        f"Task: {task_description[:300]}\n\n"
        f"Available skills:\n{index_str}"
    )

    session = BudgetedSession(
        system_prompt=_SKILL_PICKER_SYSTEM,
        tools=None,
        max_tokens=60,
        verbose=config.verbose,
    )

    matched_names: list[str] = []
    try:
        response = session.invoke(prompt)
        raw = str(response.content).strip().lower()

        if raw != "none" and raw:
            for token in re.split(r"[,\s]+", raw):
                token = token.strip()
                if token.isdigit():
                    idx = int(token) - 1
                    if 0 <= idx < len(names):
                        matched_names.append(names[idx])

    except Exception as e:
        if config.verbose:
            print(f"  [Skills] Picker failed: {e}")

    if config.verbose:
        print(f"  [Skills] Matched: {matched_names or ['(generic)']}")

    if not matched_names:
        return [], _GENERIC_SKILL

    # Merge matched skill prompts into one block
    merged = _merge_skill_prompts(matched_names, all_skills)
    return matched_names, merged


def _merge_skill_prompts(names: list[str], all_skills: dict[str, dict]) -> str:
    """Combine multiple skill prompts, truncating each to stay within budget."""
    if len(names) == 1:
        # Single skill: use full content (up to budget)
        return all_skills[names[0]]["content"][:config.llm.file_snippet_max_chars * 2]

    # Multiple skills: include header + truncated content from each
    parts = ["=== COMBINED SKILLS ==="]
    per_skill_budget = config.llm.file_snippet_max_chars // len(names)
    for name in names:
        content = all_skills[name]["content"][:per_skill_budget]
        parts.append(f"--- {name.upper()} ---\n{content}")
    return "\n\n".join(parts)


# ── Re-plan signal ────────────────────────────────────────────────────────────

def needs_resplit(matched_names: list[str]) -> bool:
    """Return True if the task matched multiple skills and should be re-planned."""
    return len(matched_names) > 1
