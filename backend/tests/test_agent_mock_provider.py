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


def test_mock_ignores_previous_turn_tool_results():
    provider = MockAgentProvider()
    tools = [
        {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}
        for name in ("global_summary", "model_list")
    ]
    response = provider.complete(ProviderRequest(
        messages=[
            ProviderMessage(role="user", content="列出数据集"),
            ProviderMessage(role="tool", content='{"tool_name":"global_summary","result":{"dataset_count":2}}'),
            ProviderMessage(role="assistant", content="数据集 2 个"),
            ProviderMessage(role="user", content="查看模型"),
        ],
        model="mock-model", tools=tools,
    ))
    assert [call.name for call in response.tool_calls] == ["model_list"]


def test_mock_fetches_evaluations_after_latest_model():
    provider = MockAgentProvider()
    tools = [
        {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}
        for name in ("model_latest_for_dataset", "evaluation_list")
    ]
    question = "请列出数据集 ds_abc 的最新模型和评估结果"
    response = provider.complete(ProviderRequest(
        messages=[
            ProviderMessage(role="user", content=question),
            ProviderMessage(role="tool", content='{"tool_name":"model_latest_for_dataset","result":{"found":true,"model":{"id":"model_abc","name":"best"}}}'),
        ], model="mock-model", tools=tools,
    ))
    assert [(call.name, call.arguments) for call in response.tool_calls] == [
        ("evaluation_list", {"model_id": "model_abc"})
    ]
