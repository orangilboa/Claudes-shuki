"""
Rules Loader

Loads user rules from:
  1. %LOCALAPPDATA%/.shuki/rules/   (global user rules)
  2. {cwd}/.shuki/                  (project-local rules)

Both locations are merged. All rules are injected into every executor call —
no per-subtask LLM filtering step.

Rule files: any .md or .txt file in the rules directories.
Each file is one rule "document". Files can be named anything descriptive,
e.g. "no_print_statements.md", "always_use_type_hints.txt".
"""
from __future__ import annotations
import os
from pathlib import Path

from config import config


# ── Rule file discovery ───────────────────────────────────────────────────────

def _rules_dirs() -> list[Path]:
    """Return all directories that may contain rule files."""
    dirs = []

    # 1. Global user rules
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    global_rules = Path(local_app_data) / ".shuki" / "rules"
    if global_rules.exists():
        dirs.append(global_rules)

    # 2. Local/Workspace rules
    roots = []
    if config.workspace.root:
        roots.append(Path(config.workspace.root))
    cwd = Path.cwd()
    if cwd not in [Path(r) for r in roots]:
        roots.append(cwd)

    for r in roots:
        shuki_dir = Path(r) / ".shuki"
        if shuki_dir.exists():
            # Check subdir first (standard)
            rules_subdir = shuki_dir / "rules"
            if rules_subdir.exists():
                dirs.append(rules_subdir)
            # Then flat (for backward compat)
            if shuki_dir not in dirs:
                dirs.append(shuki_dir)

    # Return unique paths in order
    unique = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


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


# ── Format all rules ──────────────────────────────────────────────────────────

def format_all_rules(rules: dict) -> str:
    """Format all loaded rules for injection into executor system prompt."""
    if not rules:
        return ""
    parts = ["=== RULES (must follow) ==="]
    for name, content in rules.items():
        parts.append(f"### {name}\n{content}")
    return "\n\n".join(parts)
