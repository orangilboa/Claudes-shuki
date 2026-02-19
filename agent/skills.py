"""
Skill Loader

Skills are task-type definitions — markdown/txt files that describe HOW
to approach a category of work (coding, documentation, testing, git, etc.).

Discovery order (later paths win on name collision):
  1. {install_dir}/skills/          — bundled skills shipped with Shuki
  2. %LOCALAPPDATA%/.shuki/skills/  — user-defined extensions

Each skill file is a markdown document with:
  - A short description (first non-empty line or a # header)
  - Detailed instructions for the executor
  - Optional: a list of preferred tools

The planner assigns a skill name to each subtask. The executor injects
the full skill content into its system prompt.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

from config import config


# ── Skill discovery ───────────────────────────────────────────────────────────

def _skill_dirs() -> list[Path]:
    """Return ordered list of skill directories."""
    dirs = []

    # 1. Bundled skills (relative to this file's package root)
    install_root = Path(__file__).parent.parent
    bundled_options = [
        install_root / "skills",
        install_root / ".shuki" / "skill",
        install_root / ".shuki" / "skills",
    ]
    for d in bundled_options:
        if d.exists():
            dirs.append(d)

    # 2. Global user-extended skills: %LOCALAPPDATA%\.shuki\skills
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    for sub in ["skills", "skill"]:
        user_skills = Path(local_app_data) / ".shuki" / sub
        if user_skills.exists():
            dirs.append(user_skills)

    # 3. Local/Workspace skills
    roots = []
    if config.workspace.root:
        roots.append(Path(config.workspace.root))
    cwd = Path.cwd()
    if cwd not in [Path(r) for r in roots]:
        roots.append(cwd)

    for r in roots:
        shuki_dir = Path(r) / ".shuki"
        if shuki_dir.exists():
            for sub in ["skill", "skills"]:
                subdir = shuki_dir / sub
                if subdir.exists():
                    dirs.append(subdir)

    # Return unique paths in order
    unique = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


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


# ── Public API ────────────────────────────────────────────────────────────────

_GENERIC_SKILL_CONTENT = (
    "You are a precise, careful assistant. Complete the task exactly as described. "
    "Think step by step. Prefer targeted edits over large rewrites. Verify your work."
)


def get_skill_content(name: str, all_skills: dict) -> str:
    """Return full content of a named skill, or a generic fallback."""
    if name in all_skills:
        return all_skills[name]["content"]
    return _GENERIC_SKILL_CONTENT


def build_skills_catalog(all_skills: dict) -> str:
    """Return compact index: '- coding: General software development tasks' per line."""
    if not all_skills:
        return "(no skills available — use skill: generic)"
    lines = [f"- {name}: {info['description']}" for name, info in all_skills.items()]
    return "\n".join(lines)
