from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from app.agent.planner import RuleBasedPlanner
from app.agent.provider import ProviderMessage, ProviderRequest, ProviderResponse, ProviderToolCall
from app.agent.providers.provider_planner import current_tool_results, next_evaluation_call, planner_message
from app.agent.providers.openai_compatible import DEFAULT_OPENAI_BASE, OpenAICompatibleProvider
from app.agent.reply_renderer import AgentReplyRenderer
from app.core.config import Settings
from app.core.errors import ValidationError
from app.settings.schemas import LLMSettingsInternal


SUPPORTED_PROVIDERS = frozenset({"mock", "openai", "openai-compatible", "local"})
_planner = RuleBasedPlanner()
_renderer = AgentReplyRenderer()


class MockAgentProvider:
    """Deterministic provider: RuleBasedPlanner + reply renderer."""

    name = "mock"

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        return replace(self._complete(request), source="mock")

    def _complete(self, request: ProviderRequest) -> ProviderResponse:
        tool_payloads = current_tool_results(request.messages)
        last_user = next((item.content for item in reversed(request.messages) if item.role == "user"), "")
        available = {
            item.get("function", {}).get("name")
            for item in request.tools
            if isinstance(item, dict) and item.get("function", {}).get("name")
        }
        if tool_payloads:
            follow_up = next_evaluation_call(last_user, tool_payloads, available, prefix="mock")
            if follow_up:
                return ProviderResponse(content="", tool_calls=[follow_up])
            return ProviderResponse(
                content=_renderer.format_tool_report(tool_payloads, question=last_user),
                tool_calls=[],
            )

        structured = _planner.structured_plan(planner_message(request), available=available)
        planned = [
            ProviderToolCall(name=step.tool_name, arguments=step.arguments, call_id=f"mock_{step.tool_name}")
            for step in structured.steps
        ]
        if planned:
            content = _renderer.confirmation_reply(structured) if structured.needs_confirmation else ""
            if structured.risks:
                risk_text = "；".join(structured.risks)
                content = f"{risk_text} {content}".strip() if content else risk_text
            return ProviderResponse(content=content, tool_calls=planned)

        if structured.risks:
            return ProviderResponse(
                content="；".join(structured.risks) + " 可改为查询数据集/训练/模型，或说明要创建的受管任务。",
                tool_calls=[],
            )

        preview = last_user.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        content = (
            "我是本地 Agent 助手：可查询数据集/训练/模型，也可在人工确认后提交训练、评估或自动标注任务。"
            "请提供数据集/训练/模型 ID，或说明要查询/创建的操作。"
            f" 收到：{preview or '(empty)'}"
        )
        return ProviderResponse(content=content, tool_calls=[])

class UnsupportedAgentProvider:
    """Fails explicitly when a non-supported provider is configured."""

    def __init__(self, requested: str) -> None:
        self.requested = (requested or "").strip() or "unknown"
        self.name = self.requested

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        del request
        raise ValidationError(
            "agent_provider_unsupported",
            f"Agent provider '{self.requested}' is not available. "
            "Use YWA_AGENT_PROVIDER=mock or openai-compatible.",
        )


def build_provider(provider_name: str, settings: Settings | None = None) -> Any:
    normalized = (provider_name or "mock").strip().lower() or "mock"
    if normalized == "mock":
        return MockAgentProvider()
    if normalized in {"openai", "openai-compatible", "local"}:
        if settings is None:
            return UnsupportedAgentProvider(normalized)
        effective = "openai-compatible" if normalized == "local" else normalized
        base = (settings.agent_base_url or "").strip() or (
            DEFAULT_OPENAI_BASE if effective == "openai" else ""
        )
        if not base:
            return UnsupportedAgentProvider(f"{normalized} (missing YWA_AGENT_BASE_URL)")
        if not settings.agent_api_key and "localhost" not in base and "127.0.0.1" not in base:
            return UnsupportedAgentProvider(f"{normalized} (missing YWA_AGENT_API_KEY)")
        return OpenAICompatibleProvider(
            api_key=settings.agent_api_key,
            base_url=base,
            model=settings.agent_model,
            timeout_seconds=settings.agent_timeout_seconds,
            temperature=settings.agent_temperature,
            auth_scheme=settings.agent_auth_scheme,
            auth_header_name=settings.agent_auth_header_name or "",
        )
    return UnsupportedAgentProvider(normalized)


def build_provider_from_llm(llm: LLMSettingsInternal) -> Any:
    """Resolve the provider from Settings UI LLM configuration."""
    if not llm.enabled:
        return MockAgentProvider()
    provider = (llm.provider or "openai-compatible").strip().lower() or "openai-compatible"
    if provider in {"", "mock"}:
        return MockAgentProvider()
    if provider == "local":
        provider = "openai-compatible"
    if provider not in {"openai", "openai-compatible"}:
        return UnsupportedAgentProvider(provider)
    base = (llm.api_base or "").strip() or (DEFAULT_OPENAI_BASE if provider == "openai" else "")
    if not base:
        return UnsupportedAgentProvider(f"{provider} (missing api_base)")
    if not llm.api_key and "localhost" not in base and "127.0.0.1" not in base:
        return UnsupportedAgentProvider(f"{provider} (missing api_key)")
    return OpenAICompatibleProvider(
        api_key=llm.api_key or None,
        base_url=base,
        model=llm.model or "gpt-4o-mini",
        timeout_seconds=float(llm.timeout_seconds or 60),
        temperature=float(llm.temperature),
        auth_scheme=llm.auth_scheme,
        auth_header_name=llm.auth_header_name or "",
    )


def llm_is_usable(llm: LLMSettingsInternal) -> bool:
    """Return whether persisted settings describe a callable Provider."""
    if not llm.enabled:
        return False
    provider = (llm.provider or "openai-compatible").strip().lower() or "openai-compatible"
    base = (llm.api_base or "").strip()
    if provider == "openai" and not base:
        base = DEFAULT_OPENAI_BASE
    if provider == "local":
        provider = "openai-compatible"
    if not base or not (llm.model or "").strip():
        return False
    if llm.api_key:
        return True
    if "localhost" in base or "127.0.0.1" in base:
        return True
    return provider in {"openai-compatible", "local"}


def to_provider_messages(rows: list[tuple[str, str] | ProviderMessage]) -> list[ProviderMessage]:
    messages: list[ProviderMessage] = []
    for row in rows:
        if isinstance(row, ProviderMessage):
            messages.append(row)
            continue
        role, content = row
        messages.append(ProviderMessage(role=role, content=content))
    return messages


def _extract_tool_payloads(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if message.role != "tool":
            continue
        try:
            data = json.loads(message.content)
        except json.JSONDecodeError:
            payloads.append({"tool_name": message.name or "tool", "result": {"preview": message.content[:400]}})
            continue
        if isinstance(data, dict) and "result" in data:
            payloads.append(
                {
                    "tool_name": data.get("tool_name") or message.name or "tool",
                    "result": data.get("result"),
                }
            )
        else:
            payloads.append({"tool_name": message.name or "tool", "result": data})
    return payloads


def _plan_tools(user_text: str, available: set[str]) -> list[ProviderToolCall]:
    """Compatibility wrapper used by tests; delegates to RuleBasedPlanner."""
    return _planner.plan_tool_calls(user_text, available)
