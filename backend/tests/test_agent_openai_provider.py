from __future__ import annotations

import json

import httpx
import pytest

from app.agent.llm_auth import build_llm_request_headers
from app.agent.provider import ProviderMessage, ProviderRequest
from app.agent.providers.mock import MockAgentProvider, build_provider
from app.agent.providers.openai_compatible import OpenAICompatibleProvider
from app.core.config import Settings
from app.core.errors import ValidationError


def test_build_llm_request_headers_bearer_and_custom():
    assert build_llm_request_headers("sk-test")["Authorization"] == "Bearer sk-test"
    headers = build_llm_request_headers("tok", auth_scheme="header", auth_header_name="X-Api-Key")
    assert headers["X-Api-Key"] == "tok"
    assert "Authorization" not in headers


def test_build_provider_openai_compatible(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        agent_provider="openai-compatible",
        agent_model="gpt-test",
        agent_api_key="sk-test",
        agent_base_url="http://127.0.0.1:9999/v1",
    )
    provider = build_provider("openai-compatible", settings)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "gpt-test"


def test_build_provider_rejects_unknown_and_misconfigured(tmp_path):
    assert isinstance(build_provider("mock"), MockAgentProvider)
    unsupported = build_provider("anthropic")
    with pytest.raises(ValidationError) as exc:
        unsupported.complete(ProviderRequest(messages=[ProviderMessage(role="user", content="hi")], model="x", tools=[]))
    assert exc.value.error_code == "agent_provider_unsupported"

    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        agent_provider="openai-compatible",
        agent_base_url=None,
        agent_api_key="sk",
    )
    bad = build_provider("openai-compatible", settings)
    with pytest.raises(ValidationError):
        bad.complete(ProviderRequest(messages=[ProviderMessage(role="user", content="hi")], model="x", tools=[]))


def test_openai_compatible_parses_tool_calls(monkeypatch):
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="http://example.test/v1",
        model="gpt-test",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "global_summary",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            assert url.endswith("/chat/completions")
            assert headers["Authorization"] == "Bearer sk-test"
            assert json["tools"]
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = provider.complete(
        ProviderRequest(
            messages=[ProviderMessage(role="user", content="列出数据集")],
            model="gpt-test",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "global_summary", "description": "x", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert result.tool_calls[0].name == "global_summary"
    assert result.tool_calls[0].call_id == "call_1"


def test_openai_compatible_falls_back_when_no_tool_calls(monkeypatch):
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="http://example.test/v1",
        model="gpt-test",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "I cannot help.", "tool_calls": []}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = provider.complete(
        ProviderRequest(
            messages=[ProviderMessage(role="user", content="列出有哪些数据集")],
            model="gpt-test",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "global_summary", "description": "x", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert provider.last_source == "fallback"
    assert result.tool_calls[0].name == "global_summary"


def test_openai_compatible_falls_back_on_http_error_without_exposing_response_body(monkeypatch, caplog):
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="http://example.test/v1",
        model="gpt-test",
    )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            request = httpx.Request("POST", url)
            response = httpx.Response(503, request=request, text="Authorization: Bearer sk-test")
            raise httpx.HTTPStatusError("error", request=request, response=response)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = provider.complete(
        ProviderRequest(
            messages=[ProviderMessage(role="user", content="列出有哪些数据集")],
            model="gpt-test",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "global_summary", "description": "x", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert provider.last_source == "fallback"
    assert provider.last_error == "LLM request failed: HTTP 503"
    assert "sk-test" not in "\n".join(record.getMessage() for record in caplog.records)
    assert result.tool_calls[0].name == "global_summary"
