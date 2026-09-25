from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from app.agent.profiles import ProfileId


AgentRunStatus = Literal["pending", "running", "awaiting_approval", "completed", "failed", "cancelled"]
AgentMessageRole = Literal["user", "assistant", "system", "tool"]
AgentToolCallStatus = Literal["pending", "running", "completed", "failed", "awaiting_approval"]
AgentApprovalStatus = Literal["pending", "approved", "executing", "rejected", "expired", "executed"]


class AgentSessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = Field(default=None, min_length=1, max_length=64)
    training_task_id: str | None = Field(default=None, min_length=1, max_length=64)
    model_id: str | None = Field(default=None, min_length=1, max_length=64)


class AgentMessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=16_000)
    read_only: bool = False
    context: AgentContext | None = None
    profile_id: ProfileId | None = None
    allow_context_change: bool = False


class AgentSessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    context: AgentContext = Field(default_factory=AgentContext)
    profile_id: ProfileId | None = None


class AgentProfileResponse(BaseModel):
    id: ProfileId
    version: int
    title: str
    title_en: str


class AgentMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    run_id: str | None
    role: AgentMessageRole
    content: str
    sequence: int
    created_at: datetime
    updated_at: datetime


class AgentToolCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    name: str
    arguments_json: dict[str, Any]
    result_json: dict[str, Any] | None
    status: AgentToolCallStatus
    error_message: str | None
    sequence: int
    created_at: datetime
    updated_at: datetime


class AgentApprovalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    tool_call_id: str | None
    tool_name: str
    payload_json: dict[str, Any]
    status: AgentApprovalStatus
    idempotency_key: str
    expires_at: datetime
    decided_at: datetime | None
    result_task_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class AgentInferenceStep(BaseModel):
    round: int
    source: Literal["llm", "mock", "fallback", "unknown"]
    provider: str
    model: str
    duration_ms: int
    outcome: Literal["completed", "failed", "discarded"]
    reason: str | None = None


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    status: AgentRunStatus
    provider: str
    model: str
    error_message: str | None
    stop_requested: bool
    read_only: bool
    context: AgentContext
    profile_id: ProfileId = "global"
    profile_version: int = 1
    actual_source: Literal["llm", "mock", "fallback", "mixed", "unknown"] = "unknown"
    inference_steps: list[AgentInferenceStep] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    messages: list[AgentMessageResponse] = Field(default_factory=list)
    tool_calls: list[AgentToolCallResponse] = Field(default_factory=list)
    approvals: list[AgentApprovalResponse] = Field(default_factory=list)


class AgentSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    profile_id: ProfileId = "global"
    profile_version: int = 1
    context: AgentContext = Field(default_factory=AgentContext)
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    latest_run_status: AgentRunStatus | None = None


class AgentSessionDetailResponse(AgentSessionResponse):
    messages: list[AgentMessageResponse] = Field(default_factory=list)
    runs: list[AgentRunResponse] = Field(default_factory=list)


class AgentSessionPageResponse(BaseModel):
    items: list[AgentSessionResponse]
    next_cursor: str | None = None


class AgentTimelineResponse(AgentSessionDetailResponse):
    next_before_sequence: int | None = None


class AgentProviderStatusResponse(BaseModel):
    provider: str
    model: str
    configured: bool
    api_key_configured: bool
    max_tool_rounds: int
    approval_ttl_seconds: int


class AgentApprovalDecisionResponse(BaseModel):
    approval: AgentApprovalResponse
    run: AgentRunResponse


class AgentApproveRequest(BaseModel):
    """Optional edited payload (PlanPreview-style). Always re-normalized server-side."""

    payload: dict[str, Any] | None = None
