"""
Shuki — a Claude Code-style agent for closed Windows networks.

Usage:
    python main.py                          # interactive REPL
    python main.py "add type hints to app.py"  # one-shot
    python main.py --workspace ./myproject  # specify workspace

Environment variables:
    LLM_BASE_URL     Internal LLM server (default: http://localhost:11434)
    LLM_MODEL        Model name (default: llama3)
    LLM_API_KEY      API key if required (default: ollama)
    MAX_INPUT_TOKENS   Input context budget (default: 8192)
    MAX_OUTPUT_TOKENS   Max output tokens per API call (default: 4096)
    WORKSPACE_ROOT   Working directory (default: ./workspace)
    SHUKI_SHELL      Shell: powershell or cmd (default: powershell)
    SHUKI_VERBOSE    Print debug info: 0/1 (default: 1)
"""
from __future__ import annotations
import argparse
import os
import sys
import uuid
from pathlib import Path

# Enable arrow-key history and Ctrl-R search in the REPL.
# readline is built-in on macOS/Linux; on Windows use pyreadline3 (pip install pyreadline3).
try:
    import readline
except ImportError:
    try:
        import pyreadline3 as readline  # Windows fallback
    except ImportError:
        readline = None  # history unavailable — plain input() used

# Make sure the project root and .shuki are on sys.path
root = Path(__file__).parent
sys.path.insert(0, str(root))
# Add .shuki so 'import tools' works (tools moved to .shuki/tools by setup.py)
if (root / ".shuki").exists():
    sys.path.insert(0, str(root / ".shuki"))

from config import config
from agent.state import initial_state
from agent.graph import build_graph_with_memory
from agent.session_logger import get_session_logger


BANNER = r"""
 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗  ██╗██╗   ██╗██╗  ██╗██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██║   ██║██║ ██╔╝██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗███████║██║   ██║█████╔╝ ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔══██║██║   ██║██╔═██╗ ██║
╚██████╔╝██║     ███████╗██║ ╚████║███████║██║  ██║╚██████╔╝██║  ██╗██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝
  Coding agent for closed networks
"""


def run_request(graph, request: str, thread_id: str, verbose: bool = False) -> str:
    """Run a single user request through the graph and return the final answer."""
    logger = get_session_logger()
    session_file = logger.start_session(request)
    logger.log_step(
        "SESSION",
        "START",
        f"log file: {session_file.name}",
        console=verbose,
        raw_data={"request": request, "thread_id": thread_id, "session_file": str(session_file)},
    )

    state = initial_state(request)
    thread_config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    try:
        for step in graph.stream(state, config=thread_config, stream_mode="values"):
            final_state = step

        if final_state is None:
            logger.log_step("SESSION", "ERROR", "no response from agent", console=verbose)
            return "Error: no response from agent"

        answer = final_state.get("final_answer") or "Task completed."

        plan = final_state.get("plan", [])
        done = sum(1 for t in plan if t.status == "done")
        total = len(plan)
        logger.log_step(
            "SESSION",
            "REPORT",
            f"{done}/{total} tasks completed",
            console=verbose,
            raw_data={"done": done, "total": total},
        )
        for t in plan:
            logger.log_step(
                "SESSION",
                "TASK",
                f"[{t.id}] {t.title} status={t.status}",
                console=verbose,
                raw_data=t.__dict__,
            )

        return answer
    finally:
        logger.log_step("SESSION", "END", "request finished", console=verbose)
        logger.end_session()


def interactive_repl(graph):
    """Run an interactive REPL session."""
    thread_id = str(uuid.uuid4())

    # Set up persistent readline history
    if readline is not None:
        history_file = Path.home() / ".shuki_history"
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
        import atexit
        atexit.register(readline.write_history_file, history_file)

    print(BANNER)
    print(f"Workspace : {Path(config.workspace.root).resolve()}")
    print(f"LLM       : {config.llm.model} @ {config.llm.base_url}")
    print(f"Tokens    : {config.llm.max_input_tokens} in / {config.llm.max_output_tokens} out")
    print(f"\nType your request. Use 'exit' or Ctrl-C to quit.")
    print(f"{'─'*60}\n")

    while True:
        try:
            request = input("shuki> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not request:
            continue
        if request.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        # Special commands
        if request == "!ls":
            from tools.code_tools import list_directory
            print(list_directory.invoke({"path": "."}))
            continue
        if request.startswith("!cat "):
            from tools.code_tools import read_file
            print(read_file.invoke({"path": request[5:].strip()}))
            continue

        try:
            answer = run_request(graph, request, thread_id, verbose=config.verbose)
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            if config.verbose:
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Shuki — coding agent for closed networks")
    parser.add_argument("request", nargs="?", help="One-shot request (omit for interactive mode)")
    parser.add_argument("--workspace", "-w", help="Workspace directory")
    parser.add_argument("--model", "-m", help="LLM model name")
    parser.add_argument("--url", "-u", help="LLM base URL")
    parser.add_argument("--context", "-c", type=int, help="Max input context tokens")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.workspace:
        config.workspace.root = args.workspace
    if args.model:
        config.llm.model = args.model
    if args.url:
        config.llm.base_url = args.url
    if args.context:
        config.llm.max_input_tokens = args.context
    if args.quiet:
        config.verbose = False

    # Ensure workspace exists
    ws = Path(config.workspace.root)
    ws.mkdir(parents=True, exist_ok=True)

    # Build graph
    graph = build_graph_with_memory()

    if args.request:
        # One-shot mode
        answer = run_request(graph, args.request, str(uuid.uuid4()), verbose=config.verbose)
        print(answer)
    else:
        # Interactive REPL
        interactive_repl(graph)


if __name__ == "__main__":
    main()
