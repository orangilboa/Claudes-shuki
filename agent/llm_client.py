"""
LLM client wrapper.

Supports any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM,
LocalAI, etc.) — all common self-hosted options for closed networks.
"""
from __future__ import annotations
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from config import config, LLMConfig


def build_llm(
    max_tokens: Optional[int] = None,
    temperature: float = 0.1,
    streaming: bool = False,
) -> BaseChatModel:
    cfg: LLMConfig = config.llm
    return ChatOpenAI(
        base_url=f"{cfg.base_url.rstrip('/')}/v1",
        api_key=cfg.api_key,
        model=cfg.model,
        max_tokens=max_tokens or cfg.max_output_tokens,
        temperature=temperature,
        streaming=streaming,
        max_retries=1,
    )


def build_llm_with_tools(tools: list, max_tokens: Optional[int] = None) -> BaseChatModel:
    """Build an LLM bound to a tool list for tool-calling."""
    llm = build_llm(max_tokens=max_tokens)
    return llm.bind_tools(tools)


def _strip_think_blocks(text: str) -> str:
    """
    Remove Qwen3 (and similar) <think>...</think> reasoning blocks from
    message content before storing in history.

    These blocks are internal monologue — keeping them in history inflates
    context, confuses subsequent turns, and makes the model think it already
    took actions it only *thought* about taking.
    """
    if not text:
        return text
    # Remove <think>...</think> including multiline content
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _sanitise_message(m):
    """Replace None content with empty string, strip think blocks."""
    content = m.content
    if content is None:
        content = ""
    elif isinstance(content, str):
        content = _strip_think_blocks(content)
    return m.model_copy(update={"content": content})


class BudgetedSession:
    """
    A single isolated LLM session.

    Key primitive for small-context operation: each subtask gets exactly
    ONE session with a fresh message list. Context is assembled externally
    and injected as the system prompt.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: Optional[list] = None,
        max_tokens: int = 512,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.messages: list = [SystemMessage(content=system_prompt)]
        self.tools = tools or []

        if tools:
            self.llm = build_llm_with_tools(tools, max_tokens=max_tokens)
        else:
            self.llm = build_llm(max_tokens=max_tokens)

    def invoke(self, user_message: str) -> AIMessage:
        self.messages.append(HumanMessage(content=user_message))
        if self.verbose:
            print(f"\n[Session] → {user_message}")
        response: AIMessage = self.llm.invoke(self.messages)
        # Strip think blocks before storing — they must not accumulate in history
        response = _sanitise_message(response)
        self.messages.append(response)
        if self.verbose:
            tc = len(getattr(response, "tool_calls", []) or [])
            print(f"[Session] ← {str(response.content)} [tool_calls: {tc}]")
        return response

    def append_tool_result(self, tool_call_id: str, result: str, tool_name: str):
        """Add a tool result back into the session."""
        from langchain_core.messages import ToolMessage
        self.messages.append(
            ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name)
        )

    def continue_after_tools(self) -> AIMessage:
        """
        Call the LLM again after tool results have been appended.
        Sanitises all messages before the call (None content, think blocks).
        """
        safe_messages = [_sanitise_message(m) for m in self.messages]

        if self.verbose:
            print(f"[Session] continuing ({len(safe_messages)} messages in history)")

        response: AIMessage = self.llm.invoke(safe_messages)
        response = _sanitise_message(response)
        self.messages.append(response)

        if self.verbose:
            tc = len(getattr(response, "tool_calls", []) or [])
            print(f"[Session] ← {str(response.content)} [tool_calls: {tc}]")

        return response

    @property
    def last_response(self) -> Optional[AIMessage]:
        for m in reversed(self.messages):
            if isinstance(m, AIMessage):
                return m
        return None

    def total_chars(self) -> int:
        return sum(len(str(m.content)) for m in self.messages)
