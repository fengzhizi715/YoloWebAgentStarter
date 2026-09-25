from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


ToolKind = Literal["read", "write"]
ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    kind: ToolKind
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    handler: ToolHandler | None = None
    expose_to_provider: bool = True
    canonical_name: str | None = None


class AgentToolRegistry:
    """Allowlist of Agent tools. Default deny outside this registry."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentToolSpec] = {}

    def register(self, spec: AgentToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> AgentToolSpec | None:
        return self._tools.get(name)

    def list_tools(self, *, kind: ToolKind | None = None) -> list[AgentToolSpec]:
        items = list(self._tools.values())
        if kind is None:
            return items
        return [item for item in items if item.kind == kind]

    def provider_tools(self, *, kind: ToolKind | None = None, allowed: frozenset[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI-style tool schemas for providers that support tool calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in self._tools.values()
            if tool.expose_to_provider and (kind is None or tool.kind == kind)
            and (allowed is None or (tool.canonical_name or tool.name) in allowed)
        ]


def build_default_registry(*args, **kwargs):
    """Compatibility wrapper; prefer app.agent.read_tools.build_default_registry."""
    from app.agent.read_tools import build_default_registry as _build

    return _build(*args, **kwargs)
