from __future__ import annotations

import logging
from pathlib import Path

from app.agent.redaction import REDACTED, redact_mapping, redact_text, redact_tool_arguments
from app.agent.providers.openai_compatible import OpenAICompatibleProvider
from app.agent.service import AgentService
from app.core.config import Settings
from app.core.errors import ValidationError
from app.main import create_app
from fastapi.testclient import TestClient


def test_redact_text_masks_api_key_patterns_and_known_secrets():
    secret = "sk-live-super-secret"
    text = f"Authorization: Bearer {secret} api_key={secret} normal=ok"
    redacted = redact_text(text, secrets=[secret])
    assert secret not in redacted
    assert REDACTED in redacted
    assert "normal=ok" in redacted


def test_redact_tool_arguments_masks_secret_keys():
    payload = {
        "dataset_id": "ds_abc",
        "api_key": "should-not-leak",
        "nested": {"token": "tok_123", "name": "demo"},
        "note": "api_key=embedded-secret",
    }
    redacted = redact_tool_arguments(payload, secrets=["embedded-secret"])
    assert redacted["dataset_id"] == "ds_abc"
    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["token"] == REDACTED
    assert redacted["nested"]["name"] == "demo"
    assert "embedded-secret" not in redacted["note"]
    assert REDACTED in redacted["note"]


def test_redact_mapping_preserves_non_secret_structure():
    assert redact_mapping(["a", {"password": "x"}]) == ["a", {"password": REDACTED}]


def test_agent_service_log_safe_never_emits_api_key(caplog, tmp_path):
    from app.agent.tools import AgentToolRegistry

    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'agent-log.db'}",
        agent_api_key="sk-test-should-never-appear",
    )
    service = AgentService(
        session_factory=None,  # type: ignore[arg-type]
        settings=settings,
        registry=AgentToolRegistry(),
    )
    agent_logger = logging.getLogger("ywa.agent")
    previous_disabled = agent_logger.disabled
    previous_propagate = agent_logger.propagate
    agent_logger.disabled = False
    agent_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="ywa.agent"):
            service._log_safe(
                "agent_tool_call api_key=sk-test-should-never-appear",
                arguments={"dataset_id": "ds_abc", "api_key": "sk-test-should-never-appear"},
                provider="mock",
            )
        joined = "\n".join(record.getMessage() for record in caplog.records if record.name == "ywa.agent")
        if not joined:
            joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "sk-test-should-never-appear" not in joined
        assert REDACTED in joined
        assert "ds_abc" in joined
    finally:
        agent_logger.disabled = previous_disabled
        agent_logger.propagate = previous_propagate


def test_agent_status_never_returns_api_key_value(tmp_path, monkeypatch):
    monkeypatch.setenv("YWA_AGENT_API_KEY", "sk-visible-only-as-flag")
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'status.db'}",
        agent_api_key="sk-visible-only-as-flag",
    )
    with TestClient(create_app(settings)) as client:
        payload = client.get("/api/agent/status").json()
        assert payload["api_key_configured"] is True
        assert "api_key" not in payload
        assert "sk-visible-only-as-flag" not in str(payload)


def test_agent_run_never_persists_provider_error_detail(tmp_path, monkeypatch):
    secret = "sk-provider-error-must-not-persist"
    settings = Settings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path / "data",
        import_root=tmp_path / "imports",
        database_url=f"sqlite:///{tmp_path / 'provider-error.db'}",
        agent_provider="openai-compatible",
        agent_model="mock-model",
        agent_api_key=secret,
        agent_base_url="http://provider.invalid/v1",
    )

    def fail(_self, _request):
        raise ValidationError("agent_provider_http_error", f"LLM request failed: Authorization: Bearer {secret}")

    monkeypatch.setattr(OpenAICompatibleProvider, "complete", fail)
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={}).json()["id"]
        response = client.post(f"/api/agent/sessions/{session_id}/messages", json={"content": "hello"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["error_message"] == "Agent run failed. Check local logs for details."
        assert secret not in str(body)

    db_path = Path(str(settings.database_url).removeprefix("sqlite:///"))
    assert secret.encode() not in db_path.read_bytes()
