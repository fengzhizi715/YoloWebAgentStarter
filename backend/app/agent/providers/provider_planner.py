"""Shared helpers: RuleBasedPlanner fallback for OpenAI-compatible provider."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.planner import RuleBasedPlanner
from app.agent.provider import ProviderMessage, ProviderRequest, ProviderResponse, ProviderToolCall
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


def planner_message(request: ProviderRequest) -> str:
    """Give the deterministic fallback the same explicit context as the LLM."""
    text = last_user_message(request.messages)
    if re.search(r"\b(?:ds|train|model|eval)_[\w]+", text):
        return text
    if any(word in text.lower() for word in ("所有", "全部", "列出", "list ", "all ")):
        return text
    for key, label in (("training_task_id", "训练任务"), ("model_id", "模型"), ("dataset_id", "数据集")):
        if request.context.get(key):
            return f"{text}\n当前{label}：{request.context[key]}"
    return text


def current_tool_results(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    """Only results after the latest user message belong to this request."""
    latest_user = max((index for index, item in enumerate(messages) if item.role == "user"), default=-1)
    results: list[dict[str, Any]] = []
    for item in messages[latest_user + 1:]:
        if item.role != "tool":
            continue
        try:
            value = json.loads(item.content)
        except json.JSONDecodeError:
            value = {"tool_name": item.name or "tool", "result": {"preview": item.content[:400]}}
        if isinstance(value, dict):
            results.append(value)
    return results


def has_current_tool_results(messages: list[ProviderMessage]) -> bool:
    return bool(current_tool_results(messages))


def next_evaluation_call(
    question: str, results: list[dict[str, Any]], available: set[str], *, prefix: str
) -> ProviderToolCall | None:
    if not any(token in question.lower() for token in ("评估", "evaluation", "eval")):
        return None
    if any(item.get("tool_name") in {"evaluation_list", "list_model_evaluations"} for item in results):
        return None
    tool_name = next((name for name in ("evaluation_list", "list_model_evaluations") if name in available), None)
    if not tool_name:
        return None
    for item in results:
        name = item.get("tool_name")
        result = item.get("result") or {}
        if name == "model_latest_for_dataset":
            model_id = (result.get("model") or {}).get("id")
        elif name in {"model_list", "list_models"}:
            model_id = next((row.get("id") for row in result.get("items") or [] if row.get("id")), None)
        else:
            model_id = None
        if model_id:
            return ProviderToolCall(
                name=tool_name, arguments={"model_id": model_id}, call_id=f"{prefix}_evaluation_list"
            )
    return None


def planner_fallback_response(
    request: ProviderRequest,
    *,
    reason: str | None = None,
    prefix: str = "",
) -> ProviderResponse | None:
    """Return a planner-backed response when LLM output is unusable."""
    results = current_tool_results(request.messages)
    if results:
        follow_up = next_evaluation_call(
            last_user_message(request.messages), results, available_tool_names(request.tools), prefix="fallback"
        )
        if follow_up:
            return ProviderResponse(content="", tool_calls=[follow_up])
        report = _renderer.format_tool_report(results, question=last_user_message(request.messages))
        note = f"LLM 不可用，已用本地工具结果生成报告：{reason}。\n" if reason else prefix
        return ProviderResponse(content=f"{note}{report}", tool_calls=[])
    available = available_tool_names(request.tools)
    if not available:
        return None
    user_text = planner_message(request)
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
