"""
Tools available to the Shuki executor.

Each tool is a LangChain-compatible structured tool.
Designed for Windows environments (PowerShell).
"""
from __future__ import annotations
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Annotated
from langchain_core.tools import tool

from config import config

_ws = Path(config.workspace.root).resolve()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_path(rel_or_abs: str) -> Path:
    """Resolve path, ensure it's inside the workspace."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = _ws / p
    p = p.resolve()
    # Allow read from explicitly whitelisted paths too
    allowed = [_ws] + [Path(a).resolve() for a in config.workspace.allowed_read_paths]
    if not any(str(p).startswith(str(a)) for a in allowed):
        raise PermissionError(f"Path '{p}' is outside the workspace.")
    return p


def _trunc(text: str, max_chars: int, label: str = "") -> str:
    if max_chars == 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n... [{label} truncated, {len(text)-max_chars} chars omitted] ...\n{text[-half:]}"


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def read_file(
    path: Annotated[str, "Relative path to file inside workspace"],
    start_line: Annotated[Optional[int], "First line to read (1-indexed, inclusive)"] = None,
    end_line:   Annotated[Optional[int], "Last line to read (inclusive)"] = None,
) -> str:
    """
    Read a file from the workspace. Optionally read a specific line range
    to keep context small. Returns file content as a string.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if start_line or end_line:
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        lines = lines[s:e]
        header = f"[Lines {s+1}-{min(e, len(lines)+s)} of {path}]\n"
    else:
        header = f"[Full file: {path}]\n"
    content = "".join(lines)
    return header + _trunc(content, config.llm.file_snippet_max_chars, path)


@tool
def write_file(
    path: Annotated[str, "Relative path to file inside workspace"],
    content: Annotated[str, "Full content to write"],
) -> str:
    """Write (create or overwrite) a file inside the workspace."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: Wrote {len(content)} chars to {path}"


@tool
def patch_file(
    path: Annotated[str, "Relative path to file inside workspace"],
    old_str: Annotated[str, "Exact string to find and replace (must be unique in file)"],
    new_str: Annotated[str, "Replacement string"],
) -> str:
    """
    Replace an exact substring in a file. Safer than rewriting the whole file
    when making small edits — keeps token usage low.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    original = p.read_text(encoding="utf-8", errors="replace")
    count = original.count(old_str)
    if count == 0:
        return f"ERROR: String not found in {path}. Check whitespace/indentation."
    if count > 1:
        return f"ERROR: String appears {count} times in {path}. Make old_str more unique."
    updated = original.replace(old_str, new_str, 1)
    p.write_text(updated, encoding="utf-8")
    return f"OK: Patched {path} ({len(old_str)} → {len(new_str)} chars)"


@tool
def create_file(
    path: Annotated[str, "Relative path to new file"],
    content: Annotated[str, "Initial content"] = "",
) -> str:
    """Create a new file (fails if already exists)."""
    p = _safe_path(path)
    if p.exists():
        return f"ERROR: File already exists: {path}. Use write_file to overwrite."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: Created {path}"


@tool
def delete_file(
    path: Annotated[str, "Relative path to file or directory to delete"],
) -> str:
    """Delete a file or empty directory inside the workspace."""
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: Path not found: {path}"
    if p.is_dir():
        shutil.rmtree(p)
        return f"OK: Deleted directory {path}"
    p.unlink()
    return f"OK: Deleted file {path}"


@tool
def list_directory(
    path: Annotated[str, "Relative path to directory (default: workspace root)"] = ".",
    recursive: Annotated[bool, "List recursively"] = False,
) -> str:
    """
    List files in a directory. Returns a compact tree.
    Use this to understand project structure before reading files.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: Directory not found: {path}"
    lines = []
    if recursive:
        for root, dirs, files in os.walk(p):
            # Skip hidden and common noise dirs
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'node_modules', '.git'))]
            rel = Path(root).relative_to(_ws)
            indent = "  " * len(rel.parts)
            lines.append(f"{indent}{rel}/")
            for f in sorted(files):
                lines.append(f"{indent}  {f}")
    else:
        for item in sorted(p.iterdir()):
            suffix = "/" if item.is_dir() else ""
            lines.append(f"  {item.name}{suffix}")
    return f"[Directory: {path}]\n" + "\n".join(lines[:200])


@tool
def search_in_files(
    pattern: Annotated[str, "Regex or literal string to search for"],
    path: Annotated[str, "Relative path to directory or file to search"] = ".",
    file_glob: Annotated[str, "File glob filter, e.g. '*.py'"] = "*",
    max_results: Annotated[int, "Max matching lines to return"] = 20,
) -> str:
    """
    Search for a pattern inside files. Returns matching lines with file:line context.
    Use this instead of reading full files when you need to locate something.
    """
    p = _safe_path(path)
    results = []
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: Invalid regex: {e}"

    targets = [p] if p.is_file() else p.rglob(file_glob)
    for fp in targets:
        if not fp.is_file():
            continue
        try:
            for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    rel = fp.relative_to(_ws)
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        break
        except Exception:
            continue
        if len(results) >= max_results:
            break

    if not results:
        return f"No matches for '{pattern}' in {path}"
    return "\n".join(results)


@tool
def run_command(
    command: Annotated[str, "Shell command to execute in the workspace"],
    working_dir: Annotated[str, "Working directory relative to workspace root"] = ".",
) -> str:
    """
    Run a shell command (PowerShell on Windows) inside the workspace.
    Returns stdout + stderr. Timeout enforced.
    Use for: running tests, installing packages, compiling, git operations, etc.
    """
    wd = _safe_path(working_dir)
    shell = config.workspace.shell
    timeout = config.workspace.command_timeout

    if shell == "powershell":
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        cmd = ["cmd", "/c", command]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        combined = ""
        if out:
            combined += f"STDOUT:\n{out}\n"
        if err:
            combined += f"STDERR:\n{err}\n"
        combined += f"Exit code: {result.returncode}"
        return _trunc(combined, 1200, "command output")
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_file_info(
    path: Annotated[str, "Relative path to file"],
) -> str:
    """
    Get metadata about a file (size, line count, last modified)
    without reading its content. Useful for deciding whether to read it.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: Not found: {path}"
    stat = p.stat()
    if p.is_file():
        lines = len(p.read_bytes().split(b"\n"))
        return (f"File: {path}\n"
                f"Size: {stat.st_size:,} bytes\n"
                f"Lines: {lines:,}\n"
                f"Modified: {__import__('datetime').datetime.fromtimestamp(stat.st_mtime)}")
    return f"Directory: {path} ({stat.st_size} bytes)"


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    read_file,
    write_file,
    patch_file,
    create_file,
    delete_file,
    list_directory,
    search_in_files,
    run_command,
    get_file_info,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# ── Tool classification ────────────────────────────────────────────────────────
# Used by the reasoner/writer split: reasoner gets READ_TOOLS only.

READ_TOOLS: set[str] = {
    "read_file",
    "get_file_info",
    "list_directory",
    "search_in_files",
}

WRITE_TOOLS: set[str] = {
    "write_file",
    "patch_file",
    "create_file",
    "delete_file",
    "run_command",
}
