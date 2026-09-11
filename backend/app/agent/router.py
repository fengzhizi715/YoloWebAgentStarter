from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Response
from sqlalchemy.orm import Session

from app.agent.schemas import (
    AgentApprovalDecisionResponse,
    AgentApproveRequest,
    AgentMessageCreateRequest,
    AgentProviderStatusResponse,
    AgentRunResponse,
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionResponse,
    AgentSessionUpdateRequest,
)
from app.agent.service import AgentService
from app.api.dependencies import get_agent_service, get_session


router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/status", response_model=AgentProviderStatusResponse)
def agent_provider_status(service: AgentService = Depends(get_agent_service)) -> AgentProviderStatusResponse:
    return service.provider_status()


@router.post("/sessions", response_model=AgentSessionResponse, status_code=201)
def create_agent_session(
    payload: AgentSessionCreateRequest | None = None,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentSessionResponse:
    return service.create_session(session, payload)


@router.get("/sessions", response_model=list[AgentSessionResponse])
def list_agent_sessions(
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> list[AgentSessionResponse]:
    return service.list_sessions(session)


@router.get("/sessions/{session_id}", response_model=AgentSessionDetailResponse)
def get_agent_session(
    session_id: str,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentSessionDetailResponse:
    return service.get_session(session, session_id)


@router.patch("/sessions/{session_id}", response_model=AgentSessionResponse)
def update_agent_session(
    session_id: str,
    payload: AgentSessionUpdateRequest,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentSessionResponse:
    return service.update_session(session, session_id, payload)


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_agent_session(
    session_id: str,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> Response:
    service.delete_session(session, session_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/messages", response_model=AgentRunResponse)
def post_agent_message(
    session_id: str,
    payload: AgentMessageCreateRequest,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentRunResponse:
    return service.post_message(session, session_id, payload)


@router.get("/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    run_id: str,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentRunResponse:
    return service.get_run(session, run_id)


@router.post("/runs/{run_id}/cancel", response_model=AgentRunResponse)
def cancel_agent_run(
    run_id: str,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentRunResponse:
    return service.cancel_run(session, run_id)


@router.post("/approvals/{approval_id}/approve", response_model=AgentApprovalDecisionResponse)
def approve_agent_approval(
    approval_id: str,
    body: AgentApproveRequest = Body(default_factory=AgentApproveRequest),
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentApprovalDecisionResponse:
    return service.approve_approval(session, approval_id, body.payload)


@router.post("/approvals/{approval_id}/reject", response_model=AgentApprovalDecisionResponse)
def reject_agent_approval(
    approval_id: str,
    session: Session = Depends(get_session),
    service: AgentService = Depends(get_agent_service),
) -> AgentApprovalDecisionResponse:
    return service.reject_approval(session, approval_id)
