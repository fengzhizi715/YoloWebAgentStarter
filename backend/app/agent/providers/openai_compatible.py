"""OpenAI-compatible chat + tool-calling provider.

Adapted from upstream YoloWebAgent `agent/llm_planner._chat_completion` HTTP pattern,
but wired to Starter's tool-calling Agent loop (not JSON plan mode).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.agent.llm_auth import build_llm_request_headers
from app.agent.provider import ProviderMessage, ProviderRequest, ProviderResponse, ProviderToolCall
from app.agent.providers.provider_planner import planner_fallback_response
from app.agent.redaction import redact_text
from app.core.errors import ValidationError

logger = logging.getLogger("ywa.agent")

DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


class OpenAICompatibleProvider:
    """Calls `/chat/completions` with allowlisted tools; never persists API keys."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        temperature: float = 0.2,
        auth_scheme: str = "bearer",
        auth_header_name: str = "",
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.auth_scheme = auth_scheme
        self.auth_header_name = auth_header_name
        self.last_source = "llm"
        self.last_error: str | None = None

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        try:
            response = self._complete_llm(request)
        except ValidationError as exc:
            safe_error = redact_text(exc.message, secrets=[self.api_key] if self.api_key else [])[:500]
            logger.warning("OpenAI-compatible provider failed, using planner fallback: %s", safe_error)
            self.last_source = "fallback"
            self.last_error = safe_error
            fallback = planner_fallback_response(request, reason=safe_error)
            if fallback is not None:
                return fallback
            raise

        self.last_source = "llm"
        self.last_error = None
        if response.tool_calls or not request.tools:
            return response

        fallback = planner_fallback_response(
            request,
            prefix="模型未选择工具，已改用本地规则规划。",
        )
        if fallback is not None and fallback.tool_calls:
            self.last_source = "fallback"
            self.last_error = "LLM returned no tool calls."
            return fallback
        return response

    def _complete_llm(self, request: ProviderRequest) -> ProviderResponse:
        model = (request.model or self.model).strip() or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_to_openai_message(item) for item in request.messages],
            "temperature": self.temperature,
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"

        headers = build_llm_request_headers(self.api_key, self.auth_scheme, self.auth_header_name)
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds), trust_env=False) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ValidationError("agent_provider_http_error", f"LLM request failed: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ValidationError("agent_provider_unreachable", "LLM request failed due to a network error.") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValidationError("agent_provider_bad_response", "LLM response missing choices[0].message") from exc

        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        tool_calls: list[ProviderToolCall] = []
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = _parse_arguments(function.get("arguments"))
            call_id = raw.get("id") if isinstance(raw.get("id"), str) else None
            tool_calls.append(ProviderToolCall(name=name.strip(), arguments=arguments, call_id=call_id))

        return ProviderResponse(content=content.strip(), tool_calls=tool_calls)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"value": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:500]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _to_openai_message(item: ProviderMessage) -> dict[str, Any]:
    if item.role == "tool":
        message: dict[str, Any] = {"role": "tool", "content": item.content}
        if item.tool_call_id:
            message["tool_call_id"] = item.tool_call_id
        if item.name:
            message["name"] = item.name
        return message
    return {"role": item.role, "content": item.content}
