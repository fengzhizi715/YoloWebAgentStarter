"""Shared helpers: RuleBasedPlanner fallback for OpenAI-compatible provider."""

from __future__ import annotations

from typing import Any

from app.agent.planner import RuleBasedPlanner
from app.agent.provider import ProviderRequest, ProviderResponse, ProviderToolCall
from app.agent.reply_renderer import AgentReplyRenderer

_planner = RuleBasedPlanner()
_renderer = AgentReplyRenderer()


def available_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def last_user_message(messages: list) -> str:
    for item in reversed(messages):
        if getattr(item, "role", None) == "user":
            return str(getattr(item, "content", "") or "")
    return ""


def planner_fallback_response(
    request: ProviderRequest,
    *,
    reason: str | None = None,
    prefix: str = "",
) -> ProviderResponse | None:
    """Return a planner-backed response when LLM output is unusable."""
    available = available_tool_names(request.tools)
    if not available:
        return None
    user_text = last_user_message(request.messages)
    structured = _planner.structured_plan(user_text, available=available)
    planned = [
        ProviderToolCall(name=step.tool_name, arguments=step.arguments, call_id=f"fallback_{step.tool_name}")
        for step in structured.steps
    ]
    if not planned:
        if structured.risks:
            note = "；".join(structured.risks)
            if reason:
                note = f"{note}（LLM 不可用：{reason}）"
            return ProviderResponse(content=note, tool_calls=[])
        return None

    content = _renderer.confirmation_reply(structured) if structured.needs_confirmation else ""
    if structured.risks:
        risk_text = "；".join(structured.risks)
        content = f"{risk_text} {content}".strip() if content else risk_text
    if reason:
        content = f"{prefix}LLM 不可用，已改用本地规则规划：{reason}。{content}".strip()
    elif prefix:
        content = f"{prefix}{content}".strip()
    return ProviderResponse(content=content, tool_calls=planned)
