"""
LLM client wrapper.

Supports any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM,
LocalAI, etc.) — all common self-hosted options for closed networks.
"""
from __future__ import annotations
import json
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from config import config, LLMConfig


def build_llm(
    max_tokens: Optional[int] = None,
        temperature: float = 0.1,
            streaming: bool = False,
            ) -> BaseChatModel:
                """
                    Build an LLM client pointing at the internal endpoint.

                        Works with any OpenAI-compatible server:
                              - Ollama:   base_url=http://localhost:11434/v1  model=llama3
                                    - LM Studio: base_url=http://localhost:1234/v1  model=local-model
                                          - vLLM:     base_url=http://server:8000/v1      model=model-name
                                                - OpenAI:   base_url=https://api.openai.com/v1  api_key=sk-...
                                                    """
                                                        cfg: LLMConfig = config.llm

                                                            return ChatOpenAI(
                                                                    base_url=f"{cfg.base_url.rstrip('/')}/v1",
                                                                            api_key=cfg.api_key,
                                                                                    model=cfg.model,
                                                                                            max_tokens=max_tokens or cfg.executor_budget_tokens,
                                                                                                    temperature=temperature,
                                                                                                            streaming=streaming,
                                                                                                                    # Disable retries so failures surface immediately
                                                                                                                            max_retries=1,
                                                                                                                                )


                                                                                                                                def build_llm_with_tools(tools: list, max_tokens: Optional[int] = None) -> BaseChatModel:
                                                                                                                                    """Build an LLM bound to a tool list for tool-calling."""
                                                                                                                                        llm = build_llm(max_tokens=max_tokens)
                                                                                                                                            return llm.bind_tools(tools)


                                                                                                                                            class BudgetedSession:
                                                                                                                                                """
                                                                                                                                                    A single isolated LLM session.
                                                                                                                                                        
                                                                                                                                                            The key primitive for small-context operation:
                                                                                                                                                                each subtask gets exactly ONE session with a fresh message list.
                                                                                                                                                                    Context is assembled externally and injected as the system prompt.
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
                                                                                                                                                                                                                                                                                                                        print(f"\n[Session] user → {user_message[:100]}...")
                                                                                                                                                                                                                                                                                                                                response: AIMessage = self.llm.invoke(self.messages)
                                                                                                                                                                                                                                                                                                                                        self.messages.append(response)
                                                                                                                                                                                                                                                                                                                                                if self.verbose:
                                                                                                                                                                                                                                                                                                                                                            print(f"[Session] assistant → {str(response.content)[:200]}...")
                                                                                                                                                                                                                                                                                                                                                                    return response

                                                                                                                                                                                                                                                                                                                                                                        def append_tool_result(self, tool_call_id: str, result: str, tool_name: str):
                                                                                                                                                                                                                                                                                                                                                                                """Add a tool result back into the session."""
                                                                                                                                                                                                                                                                                                                                                                                        from langchain_core.messages import ToolMessage
                                                                                                                                                                                                                                                                                                                                                                                                self.messages.append(
                                                                                                                                                                                                                                                                                                                                                                                                            ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name)
                                                                                                                                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                                                                                                                                        @property
                                                                                                                                                                                                                                                                                                                                                                                                                            def last_response(self) -> Optional[AIMessage]:
                                                                                                                                                                                                                                                                                                                                                                                                                                    for m in reversed(self.messages):
                                                                                                                                                                                                                                                                                                                                                                                                                                                if isinstance(m, AIMessage):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                return m
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return None

                                                                                                                                                                                                                                                                                                                                                                                                                                                                            def total_chars(self) -> int:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    total = 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            for m in self.messages:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        total += len(str(m.content))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                return total