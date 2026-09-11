from __future__ import annotations

from app.agent.provider import ProviderMessage, ProviderRequest, ProviderToolCall
from app.agent.providers.mock import MockAgentProvider, _plan_tools, build_provider
from app.core.errors import ValidationError
import pytest


def test_build_provider_mock_and_unsupported():
    assert isinstance(build_provider("mock"), MockAgentProvider)
    assert isinstance(build_provider(""), MockAgentProvider)
    unsupported = build_provider("anthropic")
    assert unsupported.name == "anthropic"
    with pytest.raises(ValidationError) as exc:
        unsupported.complete(
            ProviderRequest(messages=[ProviderMessage(role="user", content="hi")], model="x", tools=[])
        )
    assert exc.value.error_code == "agent_provider_unsupported"


def test_mock_complete_help_without_tools():
    provider = MockAgentProvider()
    response = provider.complete(
        ProviderRequest(
            messages=[ProviderMessage(role="user", content="hello")],
            model="mock-model",
            tools=[],
        )
    )
    assert response.tool_calls == []
    assert "Agent" in response.content or "数据集" in response.content


def test_mock_complete_plans_and_then_reports_tool_results():
    provider = MockAgentProvider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "global_summary",
                "description": "list",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    planned = provider.complete(
        ProviderRequest(
            messages=[ProviderMessage(role="user", content="列出有哪些数据集")],
            model="mock-model",
            tools=tools,
        )
    )
    assert [item.name for item in planned.tool_calls] == ["global_summary"]

    reported = provider.complete(
        ProviderRequest(
            messages=[
                ProviderMessage(role="user", content="列出有哪些数据集"),
                ProviderMessage(
                    role="tool",
                    name="global_summary",
                    content='{"tool_name":"global_summary","result":{"items":[{"id":"ds_1","name":"demo"}],"total":1}}',
                ),
            ],
            model="mock-model",
            tools=tools,
        )
    )
    assert reported.tool_calls == []
    assert "数据集列表" in reported.content


def test_mock_filters_unavailable_tools_from_plan():
    planned = _plan_tools("列出有哪些数据集", available=set())
    assert planned == []
    assert isinstance(ProviderToolCall(name="x", arguments={}, call_id="1"), ProviderToolCall)
