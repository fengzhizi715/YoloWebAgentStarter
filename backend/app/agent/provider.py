from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ProviderMessage:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ProviderToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ProviderToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ProviderRequest:
    messages: list[ProviderMessage]
    model: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    source: Literal["llm", "mock", "fallback", "unknown"] = "unknown"
    fallback_reason: str | None = None


class AgentProvider(Protocol):
    """Single LLM provider contract. Implementations must never persist API keys."""

    name: str

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Return an assistant turn, optionally requesting allowlisted tool calls."""
