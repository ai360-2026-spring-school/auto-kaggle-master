from .llm import Agent, AnthropicAgent, OfflineExpertAgent, Proposal
from .loop import ResearchLoop
from .exec_sandbox import Sandbox, ExecResult
from .tools import (ToolContext, ToolSpec, ToolResult, build_tool_registry,
                    dispatch, as_openai_tools, as_anthropic_tools,
                    as_gigachat_tools, as_yandex_tools)
from .react import ReActDriver, AssistantMessage, Message, ToolCall, Backend


def make_agent(backend: str, **kwargs):
    """Build an agent by name. Imports the SDK lazily so missing optional
    dependencies don't break offline runs."""
    backend = backend.lower()
    if backend in ("offline", "auto_offline"):
        return OfflineExpertAgent()
    if backend in ("anthropic", "anthropic-single-shot"):
        return AnthropicAgent(**kwargs)
    if backend in ("gigachat", "giga"):
        from .gigachat_agent import GigaChatAgent
        return GigaChatAgent(**kwargs)
    if backend in ("yandex", "yandexgpt", "yc", "ai-studio"):
        from .yandex_agent import YandexAgent
        return YandexAgent(**kwargs)
    raise ValueError(f"unknown agent backend: {backend!r}")


__all__ = [
    "ResearchLoop",
    "Agent", "AnthropicAgent", "OfflineExpertAgent", "Proposal",
    "Sandbox", "ExecResult",
    "ToolContext", "ToolSpec", "ToolResult",
    "build_tool_registry", "dispatch",
    "as_openai_tools", "as_anthropic_tools",
    "as_gigachat_tools", "as_yandex_tools",
    "ReActDriver", "AssistantMessage", "Message", "ToolCall", "Backend",
    "make_agent",
]
