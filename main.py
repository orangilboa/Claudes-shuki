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
    MAX_CONTEXT_TOKENS  Context window size (default: 2048)
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

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from agent.state import initial_state
from agent.graph import build_graph_with_memory


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
    state = initial_state(request)
    thread_config = {"configurable": {"thread_id": thread_id}}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Request: {request}")
        print(f"{'─'*60}")

    final_state = None
    for step in graph.stream(state, config=thread_config, stream_mode="values"):
        final_state = step

    if final_state is None:
        return "Error: no response from agent"

    answer = final_state.get("final_answer") or "Task completed."

    # Print task summary
    if verbose:
        plan = final_state.get("plan", [])
        done  = sum(1 for t in plan if t.status == "done")
        total = len(plan)
        print(f"\n{'─'*60}")
        print(f"Task report  ({done}/{total} completed):")
        STATUS_LABEL = {
            "done":          "done        ",
            "running":       "interrupted ",
            "tools":         "stuck:tools ",
            "skills":        "stuck:skills",
            "rules":         "stuck:rules ",
            "needs_resplit": "needs-resplit",
            "failed":        "FAILED      ",
            "pending":       "not reached ",
        }
        for t in plan:
            label = STATUS_LABEL.get(t.status, t.status.ljust(12))
            tools = ", ".join(c["tool"] for c in t.tool_calls_made) or "none"
            print(f"  [{label}]  [{t.id}] {t.title}")
            if t.result_summary:
                print(f"               → {t.result_summary}")
            elif t.status not in ("pending", "done"):
                print(f"               tools used: {tools}")

    return answer


def interactive_repl(graph):
    """Run an interactive REPL session."""
    thread_id = str(uuid.uuid4())
    print(BANNER)
    print(f"Workspace : {Path(config.workspace.root).resolve()}")
    print(f"LLM       : {config.llm.model} @ {config.llm.base_url}")
    print(f"Context   : {config.llm.max_context_tokens} tokens")
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
    parser.add_argument("--context", "-c", type=int, help="Max context tokens")
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
        config.llm.max_context_tokens = args.context
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
