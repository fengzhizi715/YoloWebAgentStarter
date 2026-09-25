from app.agent.provider import ProviderResponse, ProviderToolCall
from app.agent.providers.openai_compatible import OpenAICompatibleProvider
from app.agent.service import AgentService
from app.core.errors import ValidationError
from app.core.models import AgentRun


def install_provider(monkeypatch, provider):
    monkeypatch.setattr(AgentService, "_resolve_provider", lambda self: (provider, self._llm_settings()))


def test_mixed_provenance_survives_later_llm_success_and_database_reload(client, monkeypatch):
    provider = OpenAICompatibleProvider(api_key=None, base_url="http://unused.test", model="simulated")
    calls = []

    def simulated(request):
        calls.append(request)
        if len(calls) == 1:
            raise ValidationError("agent_provider_unreachable", "offline")
        return ProviderResponse(content="依据工具结果完成报告")

    monkeypatch.setattr(provider, "_complete_llm", simulated)
    install_provider(monkeypatch, provider)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(f"/api/agent/sessions/{chat}/messages?wait_for_completion=true", json={
        "content": "列出有哪些数据集", "read_only": True,
    }).json()
    assert run["status"] == "completed"
    assert run["actual_source"] == "mixed"
    assert [step["source"] for step in run["inference_steps"]] == ["fallback", "llm"]
    assert [step["round"] for step in run["inference_steps"]] == [1, 2]
    assert run["inference_steps"][0]["reason"] == "agent_provider_unreachable"
    assert run["inference_steps"][1]["reason"] is None
    assert all(step["duration_ms"] >= 0 and step["outcome"] == "completed" for step in run["inference_steps"])
    assert client.get(f"/api/agent/runs/{run['id']}").json()["inference_steps"] == run["inference_steps"]
    with client.app.state.database.session_factory() as db:
        assert db.get(AgentRun, run["id"]).inference_steps_json == run["inference_steps"]


def test_failed_inference_is_recorded_without_raw_error_or_secret(client, monkeypatch):
    provider = OpenAICompatibleProvider(api_key=None, base_url="http://unused.test", model="simulated")

    def fail(_request):
        raise RuntimeError("api_key=sk-test-private")

    monkeypatch.setattr(provider, "_complete_llm", fail)
    install_provider(monkeypatch, provider)
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{chat}/messages?wait_for_completion=true", json={"content": "hi"})
    run = response.json()
    assert run["status"] == "failed" and run["actual_source"] == "llm"
    assert run["inference_steps"][0]["outcome"] == "failed"
    assert run["inference_steps"][0]["reason"] == "provider_error"
    assert "sk-test-private" not in response.text


def test_mock_provenance_is_not_presented_as_llm(client):
    chat = client.post("/api/agent/sessions", json={}).json()["id"]
    run = client.post(f"/api/agent/sessions/{chat}/messages?wait_for_completion=true", json={"content": "hi"}).json()
    assert run["actual_source"] == "mock"
    assert run["inference_steps"][0]["source"] == "mock"
    assert run["inference_steps"][0]["reason"] is None
