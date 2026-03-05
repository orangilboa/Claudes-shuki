from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ANSI_RESET = "\033[0m"
NODE_COLORS = {
    "DISCOVERY": "\033[36m",      # cyan
    "PLANNER": "\033[31m",        # red
    "EXECUTOR": "\033[33m",       # yellow
    "VERIFIER": "\033[32m",       # green
    "SUMMARIZER": "\033[34m",     # blue
    "FINALIZER": "\033[35m",      # magenta
    "ROUTER": "\033[90m",         # bright black
    "SESSION": "\033[37m",        # white
    "LLM": "\033[96m",            # bright cyan
    "TOOL_SELECTOR": "\033[95m",  # bright magenta
}


def _humanize_seconds(seconds: int) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _slug_first_five_words(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    first_five = words[:5] or ["session"]
    return "_".join(first_five)


class SessionLogger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_file: Path | None = None
        self._waiting_thread: threading.Thread | None = None
        self._waiting_stop: threading.Event | None = None
        self._waiting_started_at: float | None = None
        self._waiting_console = False
        self._waiting_rendered = False
        self._no_color = bool(os.getenv("NO_COLOR"))

    def start_session(self, prompt: str) -> Path:
        sessions_dir = Path.cwd() / "sessions-log"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slug_first_five_words(prompt)
        session_file = sessions_dir / f"{ts}_{slug}.log"
        with self._lock:
            self._session_file = session_file
        self._write_raw(
            {
                "event": "session_start",
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
            }
        )
        return session_file

    def end_session(self) -> None:
        self.stop_waiting()
        self._write_raw({"event": "session_end", "timestamp": datetime.now().isoformat()})
        with self._lock:
            self._session_file = None

    def log_step(
        self,
        node: str,
        task: str,
        output: str,
        *,
        console: bool = True,
        raw_data: Any = None,
    ) -> None:
        node_name = node.upper().strip()
        task_name = task.upper().strip()
        line = f"[{node_name}] [{task_name}] - {output}"
        if console:
            print(self._format_console_line(node_name, line))
        payload = {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "task": task_name,
            "output": output,
            "raw_data": raw_data,
        }
        self._write_raw(payload)

    def start_waiting(self, *, console: bool = True) -> None:
        self.stop_waiting()
        stop = threading.Event()
        self._waiting_stop = stop
        self._waiting_started_at = time.time()
        self._waiting_console = console
        self._waiting_rendered = False

        def _run() -> None:
            elapsed = 0
            while not stop.wait(1):
                elapsed += 1
                text = f"waiting for llm - {_humanize_seconds(elapsed)}"
                if console:
                    line = self._format_console_line("LLM", f"[LLM] [WAIT] - {text}")
                    print(f"\r{line}", end="", flush=True)
                    self._waiting_rendered = True
                self._write_raw(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "node": "LLM",
                        "task": "WAIT",
                        "output": text,
                    }
                )

        thread = threading.Thread(target=_run, daemon=True)
        self._waiting_thread = thread
        thread.start()

    def stop_waiting(self, *, outcome: str = "success") -> None:
        thread = self._waiting_thread
        stop = self._waiting_stop
        started_at = self._waiting_started_at
        if stop:
            stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=0.2)
        self._waiting_thread = None
        self._waiting_stop = None
        self._waiting_started_at = None
        if self._waiting_console and self._waiting_rendered:
            print()
            self._waiting_rendered = False
        if started_at is not None:
            elapsed = int(time.time() - started_at)
            if outcome == "error":
                msg = f"llm call failed after {_humanize_seconds(elapsed)}"
            else:
                msg = f"llm response received after {_humanize_seconds(elapsed)}"
            self.log_step(
                "LLM",
                "WAIT",
                msg,
                console=self._waiting_console,
            )

    def _format_console_line(self, node_name: str, line: str) -> str:
        color = NODE_COLORS.get(node_name, "")
        if color and not self._no_color:
            node_token = f"[{node_name}]"
            if line.startswith(node_token):
                return f"{color}{node_token}{ANSI_RESET}{line[len(node_token):]}"
        return line

    def _write_raw(self, payload: dict[str, Any]) -> None:
        with self._lock:
            target = self._session_file
        if not target:
            return
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_LOGGER = SessionLogger()


def get_session_logger() -> SessionLogger:
    return _LOGGER
