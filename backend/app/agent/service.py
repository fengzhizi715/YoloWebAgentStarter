from __future__ import annotations

import json
import logging
from datetime import timedelta
from time import monotonic
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.agent.bounds import bound_json, untrusted_text
from app.agent.provider import ProviderMessage, ProviderRequest, ProviderToolCall
from app.agent.prompt import SYSTEM_PROMPT, bounded_history
from app.agent.profiles import PROFILES, get_profile, infer_profile, permits
from app.agent.providers.mock import build_provider, build_provider_from_llm, llm_is_usable
from app.agent.read_tools import build_default_registry
from app.agent.redaction import redact_text, redact_tool_arguments
from app.agent.schemas import (
    AgentApprovalDecisionResponse,
    AgentApprovalResponse,
    AgentContext,
    AgentMessageCreateRequest,
    AgentMessageResponse,
    AgentProviderStatusResponse,
    AgentProfileResponse,
    AgentRunResponse,
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionResponse,
    AgentSessionUpdateRequest,
    AgentToolCallResponse,
)
from app.agent.tools import AgentToolRegistry
from app.agent.write_tools import WRITE_TOOL_NAMES, normalize_write_payload, preflight_write_payload, reserved_id_prefix
from app.auto_annotation.schemas import AutoAnnotationCreateRequest
from app.auto_annotation.service import AutoAnnotationService
from app.core.config import Settings
from app.core.errors import ConflictError, DomainError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.settings.service import SettingsService
from app.core.models import (
    AgentApproval,
    AgentMessage,
    AgentRun,
    AgentSession,
    AgentToolCall,
    AutoAnnotationTask,
    Dataset,
    ModelEvaluationRecord,
    ModelVersion,
    TrainingTask,
)
from app.core.storage import Storage
from app.core.time import utc_now
from app.models.evaluation import YoloEvaluationRunner
from app.models.service import ModelService
from app.training.schemas import TrainingTaskCreate
from app.training.service import TrainingService


logger = logging.getLogger("ywa.agent")

_DEFAULT_SESSION_TITLES = frozenset({"New chat", "新会话"})


def message_response(row: AgentMessage) -> AgentMessageResponse:
    return AgentMessageResponse.model_validate(row)


def tool_call_response(row: AgentToolCall) -> AgentToolCallResponse:
    return AgentToolCallResponse.model_validate(row)


def approval_response(row: AgentApproval) -> AgentApprovalResponse:
    return AgentApprovalResponse.model_validate(row)


def run_response(row: AgentRun) -> AgentRunResponse:
    steps = row.inference_steps_json or []
    sources = {step["source"] for step in steps}
    return AgentRunResponse(
        id=row.id,
        session_id=row.session_id,
        status=row.status,  # type: ignore[arg-type]
        provider=row.provider,
        model=row.model,
        error_message=row.error_message,
        stop_requested=row.stop_requested,
        read_only=row.read_only,
        context=AgentContext.model_validate(row.context_json or {}),
        profile_id=row.profile_id,
        profile_version=row.profile_version,
        actual_source=next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unknown"),
        inference_steps=steps,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        messages=[message_response(item) for item in sorted(row.messages, key=lambda m: m.sequence)],
        tool_calls=[tool_call_response(item) for item in sorted(row.tool_calls, key=lambda t: t.sequence)],
        approvals=[approval_response(item) for item in row.approvals],
    )


class AgentService:
    """Persists Agent sessions/runs, read tools, and human-confirmed write tools."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: Settings,
        registry: AgentToolRegistry | None = None,
        storage: Storage | None = None,
        *,
        training_service: TrainingService | None = None,
        model_service: ModelService | None = None,
        auto_annotation_service: AutoAnnotationService | None = None,
        evaluation_runner: YoloEvaluationRunner | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.storage = storage
        self.registry = registry or build_default_registry(storage, session_factory=session_factory)
        self.training_service = training_service
        self.model_service = model_service
        self.auto_annotation_service = auto_annotation_service
        self.evaluation_runner = evaluation_runner

    def _llm_settings(self):
        return SettingsService(self.settings).get_llm_settings_internal()

    def _resolve_provider(self):
        llm = self._llm_settings()
        if llm_is_usable(llm):
            return build_provider_from_llm(llm), llm
        return build_provider("mock"), llm

    def _secret_values(self) -> list[str]:
        secrets: list[str] = []
        if self.settings.agent_api_key:
            secrets.append(self.settings.agent_api_key)
        try:
            key = self._llm_settings().api_key
        except Exception:  # noqa: BLE001
            key = ""
        if key and key not in secrets:
            secrets.append(key)
        return secrets

    def _log_safe(self, message: str, **fields: Any) -> None:
        safe_fields = {
            key: redact_tool_arguments(value, secrets=self._secret_values())
            if isinstance(value, dict)
            else redact_text(value, secrets=self._secret_values())
            for key, value in fields.items()
        }
        logger.info("%s %s", redact_text(message, secrets=self._secret_values()), safe_fields)

    def provider_status(self) -> AgentProviderStatusResponse:
        provider, llm = self._resolve_provider()
        configured = provider.name in {"mock", "openai-compatible"}
        display_provider = provider.name
        display_model = "mock-model" if provider.name == "mock" else (llm.model or self.settings.agent_model)
        return AgentProviderStatusResponse(
            provider=display_provider,
            model=display_model,
            configured=bool(configured),
            api_key_configured=bool(llm.api_key),
            max_tool_rounds=self.settings.agent_max_tool_rounds,
            approval_ttl_seconds=self.settings.agent_approval_ttl_seconds,
        )

    def create_session(self, session: Session, payload: AgentSessionCreateRequest | None = None) -> AgentSessionResponse:
        payload = payload or AgentSessionCreateRequest()
        context = self._validate_context(session, payload.context)
        profile = get_profile(payload.profile_id or infer_profile(context))
        title = (payload.title if payload and payload.title else None) or "New chat"
        row = AgentSession(id=new_id("asess"), title=title.strip() or "New chat",
                           profile_id=profile.id, profile_version=profile.version, context_json=context)
        session.add(row)
        session.commit()
        session.refresh(row)
        return AgentSessionResponse(
            id=row.id,
            title=row.title,
            profile_id=row.profile_id, profile_version=row.profile_version,
            context=AgentContext.model_validate(row.context_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
            message_count=0,
            latest_run_status=None,
        )

    def list_profiles(self) -> list[AgentProfileResponse]:
        return [AgentProfileResponse(id=p.id, version=p.version, title=p.title, title_en=p.title_en)
                for p in PROFILES.values()]

    def list_sessions(self, session: Session) -> list[AgentSessionResponse]:
        rows = list(session.scalars(select(AgentSession).order_by(AgentSession.updated_at.desc(), AgentSession.id.desc())))
        return [self._session_summary(session, row) for row in rows]

    def get_session(self, session: Session, session_id: str) -> AgentSessionDetailResponse:
        row = self._get_session(session, session_id)
        messages = list(
            session.scalars(
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id)
                .order_by(AgentMessage.sequence.asc(), AgentMessage.id.asc())
            )
        )
        runs = list(
            session.scalars(
                select(AgentRun)
                .where(AgentRun.session_id == session_id)
                .options(
                    selectinload(AgentRun.messages),
                    selectinload(AgentRun.tool_calls),
                    selectinload(AgentRun.approvals),
                )
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
            )
        )
        summary = self._session_summary(session, row)
        return AgentSessionDetailResponse(
            **summary.model_dump(),
            messages=[message_response(item) for item in messages],
            runs=[run_response(item) for item in runs],
        )

    def update_session(self, session: Session, session_id: str, payload: AgentSessionUpdateRequest) -> AgentSessionResponse:
        row = self._get_session(session, session_id)
        title = payload.title.strip()
        if not title:
            raise ValidationError("invalid_agent_session_title", "Session title must not be empty.")
        row.title = title
        session.commit()
        session.refresh(row)
        return self._session_summary(session, row)

    def delete_session(self, session: Session, session_id: str) -> None:
        # Serialize deletion with submissions before inspecting active runs.
        session.execute(update(AgentSession).where(AgentSession.id == session_id).values(updated_at=utc_now()))
        row = self._get_session(session, session_id)
        active = session.scalar(select(AgentRun.id).where(
            AgentRun.session_id == session_id, AgentRun.status.in_(("pending", "running", "awaiting_approval"))
        ))
        if active:
            raise ConflictError("agent_run_in_progress", "Cancel the active run before deleting its session.")
        session.delete(row)
        session.commit()

    def _validate_context(self, session: Session, context: AgentContext) -> dict[str, str]:
        result = context.model_dump(exclude_none=True)
        dataset_id = context.dataset_id
        for key, model in (("training_task_id", TrainingTask), ("model_id", ModelVersion)):
            if key not in result:
                continue
            entity = session.get(model, result[key])
            if entity is None:
                raise ValidationError("agent_context_not_found", f"Context {key} was not found.")
            owner = entity.dataset_id
            if owner and dataset_id and owner != dataset_id:
                raise ValidationError("agent_context_mismatch", "Context entities belong to different datasets.")
            dataset_id = dataset_id or owner
        if dataset_id:
            if session.get(Dataset, dataset_id) is None:
                raise ValidationError("agent_context_not_found", "Context dataset was not found.")
            result["dataset_id"] = dataset_id
        return result

    def post_message(
        self, session: Session, session_id: str, payload: AgentMessageCreateRequest, *, execute: bool = True,
        profile_version: int | None = None,
    ) -> AgentRunResponse:
        content = payload.content.strip()
        if not content:
            raise ValidationError("invalid_agent_message", "Message content must not be empty.")
        # Serialize submissions for this session before checking its active run.
        session.execute(update(AgentSession).where(AgentSession.id == session_id).values(updated_at=utc_now()))
        chat = self._get_session(session, session_id)
        session.refresh(chat)
        context = self._validate_context(session, payload.context if payload.context is not None
                                         else AgentContext.model_validate(chat.context_json or {}))
        # Context and mode are independent bindings. Re-sending (or changing)
        # context must not silently replace the mode selected for this session.
        profile_id = payload.profile_id or chat.profile_id
        version = profile_version if profile_version is not None else (
            chat.profile_version if profile_id == chat.profile_id else get_profile(profile_id).version
        )
        profile = get_profile(profile_id, version)

        active = session.scalar(
            select(AgentRun).where(
                AgentRun.session_id == session_id,
                AgentRun.status.in_(("pending", "running", "awaiting_approval")),
            )
        )
        if active is not None:
            raise ConflictError("agent_run_in_progress", "Wait for the current run to finish, or cancel it first.")

        next_sequence = self._next_message_sequence(session, session_id)
        if next_sequence > 1 and not payload.allow_context_change and (
            context != (chat.context_json or {}) or profile.id != chat.profile_id or profile.version != chat.profile_version
        ):
            raise ConflictError("agent_session_binding_changed", "Assistant mode or context changed. Start a new session or explicitly continue this one.")
        chat.context_json = context
        chat.profile_id = profile.id
        chat.profile_version = profile.version
        provider, llm = self._resolve_provider()
        run_provider = "mock" if provider.name == "mock" else (llm.provider or provider.name)
        run_model = "mock-model" if provider.name == "mock" else (llm.model or self.settings.agent_model)
        run = AgentRun(
            id=new_id("arun"),
            session_id=session_id,
            status="pending",
            provider=run_provider,
            model=run_model,
            read_only=payload.read_only,
            context_json=context,
            profile_id=profile.id,
            profile_version=profile.version,
        )
        session.add(run)
        session.flush()

        user_message = AgentMessage(
            id=new_id("amsg"),
            session_id=session_id,
            run_id=run.id,
            role="user",
            content=content,
            sequence=next_sequence,
        )
        session.add(user_message)
        if chat.title in _DEFAULT_SESSION_TITLES:
            chat.title = content[:80]
        session.commit()

        return self._execute_run(session, run.id) if execute else self.get_run(session, run.id)

    def retry_run(self, session: Session, run_id: str, *, execute: bool = True) -> AgentRunResponse:
        original = self._get_run(session, run_id)
        if original.status not in {"failed", "cancelled"}:
            raise ConflictError("agent_run_not_retryable", "Only failed or cancelled runs can be retried.")
        message = next((item for item in original.messages if item.role == "user"), None)
        if message is None:
            raise ConflictError("agent_run_not_retryable", "The original question is unavailable.")
        return self.post_message(session, original.session_id, AgentMessageCreateRequest(
            content=message.content, read_only=original.read_only,
            context=AgentContext.model_validate(original.context_json or {}),
            profile_id=original.profile_id, allow_context_change=True,
        ), execute=execute, profile_version=original.profile_version)

    def execute_run(self, run_id: str) -> None:
        """A worker owns its session; never reuse the HTTP request's session."""
        with self.session_factory() as session:
            try:
                self._execute_run(session, run_id)
            except NotFoundError:
                # A cancelled session may have been deleted before its worker resumed.
                return

    def get_run(self, session: Session, run_id: str) -> AgentRunResponse:
        run = self._get_run(session, run_id)
        return run_response(run)

    def cancel_run(self, session: Session, run_id: str) -> AgentRunResponse:
        # Read status/approvals only after acquiring the same write lock used by
        # worker finalization, so cancellation cannot overwrite a completed run
        # or miss an approval committed while it was waiting for the lock.
        session.execute(update(AgentRun).where(AgentRun.id == run_id).values(updated_at=utc_now()))
        run = self._get_run(session, run_id)
        if run.status in {"completed", "failed", "cancelled"}:
            return run_response(run)
        if any(approval.status == "executing" for approval in run.approvals):
            raise ConflictError(
                "agent_approval_execution_in_progress",
                "A confirmed operation is already being executed and cannot be cancelled.",
            )
        run.stop_requested = True
        now = utc_now()
        if run.status in {"pending", "running", "awaiting_approval"}:
            run.status = "cancelled"
            run.finished_at = now
            run.error_message = run.error_message or "Cancelled by user."
            for approval in run.approvals:
                if approval.status in {"pending", "approved"}:
                    approval.status = "rejected"
                    approval.decided_at = now
                    approval.error_message = "Cancelled by user."
            for tool in run.tool_calls:
                if tool.status in {"pending", "running", "awaiting_approval"}:
                    tool.status = "failed"
                    tool.error_message = "Cancelled by user."
        session.commit()
        return self.get_run(session, run_id)

    def approve_approval(
        self,
        session: Session,
        approval_id: str,
        payload_override: dict[str, Any] | None = None,
    ) -> AgentApprovalDecisionResponse:
        approval = self._get_approval(session, approval_id)
        run = self._get_run(session, approval.run_id)
        now = utc_now()

        if approval.status == "executed" and approval.result_task_id:
            return AgentApprovalDecisionResponse(approval=approval_response(approval), run=run_response(run))
        if run.read_only or not permits(get_profile(run.profile_id, run.profile_version), self.registry.get(approval.tool_name)):
            raise ConflictError("agent_profile_tool_not_allowed", "This run's assistant mode does not permit this operation.")
        if approval.status == "rejected":
            raise ConflictError("agent_approval_rejected", "This approval was already rejected.")
        if approval.status == "expired" or (approval.status == "pending" and approval.expires_at <= now):
            if approval.status != "expired":
                approval.status = "expired"
                session.commit()
            raise ConflictError("agent_approval_expired", "This approval has expired.")
        if approval.status not in {"pending", "approved", "executing"}:
            raise ConflictError("agent_approval_not_pending", "Only pending approvals can be confirmed.")
        if approval.status == "pending" and run.status != "awaiting_approval":
            raise ConflictError("agent_run_not_awaiting_approval", "The related run is not awaiting approval.")

        # Claim and reserve the managed task ID together. The conditional update
        # gives exactly one pending confirmation request ownership of the task.
        if approval.status == "pending":
            try:
                base = dict(approval.payload_json or {})
                if payload_override is not None:
                    if not isinstance(payload_override, dict):
                        raise ValidationError("invalid_agent_write_payload", "payload must be an object.")
                    merged = {**base, **dict(payload_override)}
                    for meta in ("action", "preview", "reserved_task_id", "requires_approval", "approval_id"):
                        merged.pop(meta, None)
                    cleaned = normalize_write_payload(approval.tool_name, merged)
                    if self.storage is not None:
                        preflight_write_payload(
                            session,
                            self.storage,
                            approval.tool_name,
                            cleaned,
                            training_service=self.training_service,
                            model_service=self.model_service,
                            session_factory=self.session_factory,
                        )
                else:
                    cleaned = normalize_write_payload(approval.tool_name, base)
            except DomainError:
                raise
            reserved = new_id(reserved_id_prefix(approval.tool_name))
            claimed_payload = {**cleaned, "reserved_task_id": reserved}
            claimed = session.execute(
                update(AgentApproval)
                .where(
                    AgentApproval.id == approval_id,
                    AgentApproval.status == "pending",
                    AgentApproval.expires_at > now,
                )
                .values(
                    payload_json=claimed_payload,
                    status="approved",
                    decided_at=now,
                    result_task_id=reserved,
                )
            )
            session.commit()
            if claimed.rowcount != 1:
                session.expire_all()
                current = self._get_approval(session, approval_id)
                current_run = self._get_run(session, current.run_id)
                if current.status == "executed" and current.result_task_id:
                    return AgentApprovalDecisionResponse(
                        approval=approval_response(current), run=run_response(current_run)
                    )
                raise ConflictError(
                    "agent_approval_already_claimed",
                    "This approval is already being processed by another confirmation request.",
                )
            approval = self._get_approval(session, approval_id)
            run = self._get_run(session, approval.run_id)
        elif payload_override is not None:
            raise ConflictError(
                "agent_approval_already_claimed",
                "This approval was already claimed; payload can only be edited while pending.",
            )

        if approval.result_task_id and self._managed_task_exists(session, approval.tool_name, approval.result_task_id):
            return self._finalize_executed_approval(session, approval_id, approval.result_task_id, now)

        if not approval.result_task_id:
            raise ConflictError("agent_approval_missing_task_id", "The approval has no reserved managed task ID.")

        # An approved record may be a retry after a process restart. Only the
        # request that atomically advances it to executing may call a service.
        executing = session.execute(
            update(AgentApproval)
            .where(AgentApproval.id == approval_id, AgentApproval.status == "approved")
            .values(status="executing", error_message=None)
        )
        session.commit()
        if executing.rowcount != 1:
            session.expire_all()
            current = self._get_approval(session, approval_id)
            current_run = self._get_run(session, current.run_id)
            if current.status == "executed" and current.result_task_id:
                return AgentApprovalDecisionResponse(approval=approval_response(current), run=run_response(current_run))
            raise ConflictError(
                "agent_approval_execution_in_progress",
                "This approval is already being executed by another request.",
            )
        approval = self._get_approval(session, approval_id)

        try:
            task_id, summary = self._execute_approved_payload(session, approval, reserved_task_id=approval.result_task_id)
        except DomainError as exc:
            # No service was successfully finalized; leave the stable reservation
            # available for a safe retry with the same managed task ID.
            approval = self._get_approval(session, approval_id)
            approval.status = "approved"
            approval.error_message = redact_text(exc.message, secrets=self._secret_values())[:500]
            session.commit()
            raise

        return self._finalize_executed_approval(session, approval_id, task_id, now, summary=summary)

    def _finalize_executed_approval(
        self,
        session: Session,
        approval_id: str,
        task_id: str,
        now,
        *,
        summary: str | None = None,
    ) -> AgentApprovalDecisionResponse:
        approval = self._get_approval(session, approval_id)
        run = self._get_run(session, approval.run_id)
        if approval.status == "executed" and approval.result_task_id:
            return AgentApprovalDecisionResponse(approval=approval_response(approval), run=run_response(run))

        approval.status = "executed"
        approval.result_task_id = task_id
        approval.decided_at = approval.decided_at or now
        approval.error_message = None
        if approval.tool_call_id:
            tool = session.get(AgentToolCall, approval.tool_call_id)
            if tool is not None:
                tool.status = "completed"
                tool.result_json = {
                    "approved": True,
                    "task_id": task_id,
                    "tool_name": approval.tool_name,
                    "payload": approval.payload_json,
                }
                tool.error_message = None
        text = summary or (
            f"已确认并创建任务 `{task_id}`（`{approval.tool_name}`）。"
            "本轮结束，不会自动串联下一步。"
        )
        if not any(item.role == "assistant" and task_id in (item.content or "") for item in run.messages):
            assistant = AgentMessage(
                id=new_id("amsg"),
                session_id=run.session_id,
                run_id=run.id,
                role="assistant",
                content=text,
                sequence=self._next_message_sequence(session, run.session_id),
            )
            session.add(assistant)
            run.messages.append(assistant)
        run.status = "completed"
        run.finished_at = now
        session.commit()
        self._log_safe(
            "agent_approval_executed",
            approval_id=approval_id,
            tool_name=approval.tool_name,
            result_task_id=task_id,
            payload=approval.payload_json or {},
        )
        run = self._get_run(session, run.id)
        approval = self._get_approval(session, approval_id)
        return AgentApprovalDecisionResponse(approval=approval_response(approval), run=run_response(run))

    def reject_approval(self, session: Session, approval_id: str) -> AgentApprovalDecisionResponse:
        approval = self._get_approval(session, approval_id)
        run = self._get_run(session, approval.run_id)
        now = utc_now()
        if approval.status == "rejected":
            return AgentApprovalDecisionResponse(approval=approval_response(approval), run=run_response(run))
        if approval.status == "executed":
            raise ConflictError("agent_approval_already_executed", "This approval was already executed.")
        if approval.status == "expired" or (approval.status == "pending" and approval.expires_at <= now):
            approval.status = "expired"
            session.commit()
            raise ConflictError("agent_approval_expired", "This approval has expired.")
        if approval.status in {"approved", "executing"} and approval.result_task_id and self._managed_task_exists(
            session, approval.tool_name, approval.result_task_id
        ):
            raise ConflictError("agent_approval_already_executed", "This approval already created a managed task.")
        if approval.status == "executing":
            raise ConflictError("agent_approval_execution_in_progress", "This approval is already being executed.")
        if approval.status not in {"pending", "approved"}:
            raise ConflictError("agent_approval_not_pending", "Only pending approvals can be rejected.")

        approval.status = "rejected"
        approval.decided_at = now
        if approval.tool_call_id:
            tool = session.get(AgentToolCall, approval.tool_call_id)
            if tool is not None:
                tool.status = "failed"
                tool.error_message = "Rejected by user."
        assistant = AgentMessage(
            id=new_id("amsg"),
            session_id=run.session_id,
            run_id=run.id,
            role="assistant",
            content=f"已拒绝操作 `{approval.tool_name}`，未创建任何任务。",
            sequence=self._next_message_sequence(session, run.session_id),
        )
        session.add(assistant)
        run.messages.append(assistant)
        run.status = "cancelled"
        run.finished_at = now
        run.error_message = "Approval rejected by user."
        session.commit()
        run = self._get_run(session, run.id)
        approval = self._get_approval(session, approval_id)
        return AgentApprovalDecisionResponse(approval=approval_response(approval), run=run_response(run))

    def recover_orphaned(self) -> None:
        """Mark interrupted runs as failed and make claimed approvals retryable."""
        with self.session_factory() as session:
            runs = list(session.scalars(select(AgentRun).where(AgentRun.status.in_(("pending", "running")))))
            executing_approvals = list(
                session.scalars(select(AgentApproval).where(AgentApproval.status == "executing"))
            )
            now = utc_now()
            for run in runs:
                run.status = "failed"
                run.error_message = "Agent run was interrupted by a service restart."
                run.finished_at = now
            for approval in executing_approvals:
                approval.status = "approved"
                approval.error_message = "Execution was interrupted by a service restart; confirmation can be retried."
            if runs or executing_approvals:
                session.commit()

    def _execute_run(self, session: Session, run_id: str) -> AgentRunResponse:
        run = self._get_run(session, run_id)
        started = session.execute(update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status == "pending", AgentRun.stop_requested.is_(False),
        ).values(status="running", started_at=utc_now()))
        session.commit()
        if started.rowcount != 1:
            return self.get_run(session, run_id)
        self._log_safe("agent_run_started", run_id=run_id, session_id=run.session_id, provider=run.provider)

        try:
            for _round in range(self.settings.agent_max_tool_rounds):
                run = self._get_run(session, run_id)
                if run.stop_requested:
                    return self._finish_run(session, run, status="cancelled", error="Cancelled by user.")

                request = ProviderRequest(
                    messages=self._provider_history(session, run.session_id, run.id),
                    model=run.model,
                    tools=self.registry.provider_tools(kind="read" if run.read_only else None,
                        allowed=get_profile(run.profile_id, run.profile_version).allowed_tools),
                    context=run.context_json or {},
                )
                session.commit()  # Do not hold a database transaction during inference.
                provider, _llm = self._resolve_provider()
                started = monotonic()
                try:
                    response = provider.complete(request)
                except Exception as exc:
                    self._record_inference_step(
                        session, run_id, provider=provider.name, model=request.model,
                        source=getattr(provider, "last_source", "mock" if provider.name == "mock" else "unknown"),
                        duration_ms=int((monotonic() - started) * 1000), outcome="failed",
                        reason=getattr(exc, "error_code", "provider_error"),
                    )
                    raise
                self._record_inference_step(
                    session, run_id, provider=provider.name, model=request.model,
                    source="mock" if response.source == "unknown" and provider.name == "mock" else response.source,
                    duration_ms=int((monotonic() - started) * 1000), outcome="completed",
                    reason=response.fallback_reason,
                )

                run = self._get_run(session, run_id)
                if run.stop_requested:
                    return self._finish_run(session, run, status="cancelled", error="Cancelled by user.")

                if response.tool_calls:
                    awaiting = self._execute_tool_calls(session, run, response.tool_calls, read_only=run.read_only)
                    if awaiting:
                        run = self._get_run(session, run_id)
                        if not self._claim_running_step(session, run_id):
                            session.rollback()
                            return self.get_run(session, run_id)
                        assistant = AgentMessage(
                            id=new_id("amsg"),
                            session_id=run.session_id,
                            run_id=run.id,
                            role="assistant",
                            content=response.content.strip()
                            if (response.content or "").strip()
                            else "已生成待执行操作，请在确认卡片中审核参数后再执行。确认后本轮结束，不会自动串联下一步。",
                            sequence=self._next_message_sequence(session, run.session_id),
                        )
                        session.add(assistant)
                        run.messages.append(assistant)
                        run.status = "awaiting_approval"
                        session.commit()
                        return self.get_run(session, run_id)
                    continue

                content = (response.content or "").strip() or "已完成，但模型没有返回可读内容。"
                assistant = AgentMessage(
                    id=new_id("amsg"),
                    session_id=run.session_id,
                    run_id=run.id,
                    role="assistant",
                    content=content,
                    sequence=self._next_message_sequence(session, run.session_id),
                )
                session.add(assistant)
                run.messages.append(assistant)
                return self._finish_run(session, run, status="completed")

            run = self._get_run(session, run_id)
            return self._finish_run(
                session,
                run,
                status="failed",
                error=f"Exceeded max tool rounds ({self.settings.agent_max_tool_rounds}).",
            )
        except Exception as exc:  # noqa: BLE001 - provider/tool failures become run failures
            session.rollback()
            run = self._get_run(session, run_id)
            if run.stop_requested:
                return self._finish_run(session, run, status="cancelled", error="Cancelled by user.")
            self._log_safe("agent_run_failed", run_id=run_id, error=str(exc)[:2000])
            return self._finish_run(session, run, status="failed", error="Agent run failed. Check local logs for details.")

    def _execute_tool_calls(
        self, session: Session, run: AgentRun, tool_calls: list[ProviderToolCall], *, read_only: bool = False
    ) -> bool:
        """Execute tool calls. Returns True when the run should pause for approval."""
        next_tool_sequence = max((item.sequence for item in run.tool_calls), default=0) + 1
        awaiting_approval = False
        for index, call in enumerate(tool_calls):
            if not self._claim_running_step(session, run.id):
                session.rollback()
                return False
            spec = self.registry.get(call.name)
            arguments = call.arguments if isinstance(call.arguments, dict) else {}
            row = AgentToolCall(
                id=new_id("atool"),
                run_id=run.id,
                name=call.name,
                arguments_json=arguments,
                status="pending",
                sequence=next_tool_sequence + index,
            )
            session.add(row)
            run.tool_calls.append(row)
            session.flush()
            self._log_safe(
                "agent_tool_call",
                run_id=run.id,
                tool_name=call.name,
                kind=None if spec is None else spec.kind,
                arguments=arguments,
            )

            if spec is None:
                result = {"error": "tool_not_allowed", "message": f"Tool '{call.name}' is not in the allowlist."}
                row.status = "failed"
                row.error_message = result["message"]
                row.result_json = result
            elif not permits(get_profile(run.profile_id, run.profile_version), spec):
                result = {"error": "agent_profile_tool_not_allowed", "message": "This tool is outside the current assistant mode. Switch mode explicitly to continue."}
                row.status = "failed"
                row.error_message = result["message"]
                row.result_json = result
            elif read_only and spec.kind == "write":
                result = {"error": "read_only_run", "message": "This question is read-only; no task can be created."}
                row.status = "failed"
                row.error_message = result["message"]
                row.result_json = result
            elif spec.handler is None:
                result = {"error": "tool_handler_missing", "message": f"Tool '{call.name}' has no handler."}
                row.status = "failed"
                row.error_message = result["message"]
                row.result_json = result
            elif spec.kind == "write":
                row.status = "running"
                session.commit()
                try:
                    payload = bound_json(spec.handler(session, arguments))
                    if not isinstance(payload, dict):
                        payload = {"value": payload}
                    row = session.get(AgentToolCall, row.id) or row
                    run = self._get_run(session, run.id)
                    if not self._claim_running_step(session, run.id):
                        row.status = "failed"
                        row.error_message = "Cancelled by user."
                        session.commit()
                        return False
                    approval = self.create_approval_record(
                        session,
                        run=run,
                        tool_name=call.name,
                        payload_json=payload,
                        tool_call_id=row.id,
                        idempotency_key=f"{run.id}:{row.id}",
                    )
                    row.status = "awaiting_approval"
                    row.result_json = {
                        "requires_approval": True,
                        "approval_id": approval.id,
                        "preview": payload.get("preview") or payload,
                    }
                    awaiting_approval = True
                except DomainError as exc:
                    result = {"error": exc.error_code, "message": untrusted_text(exc.message, limit=400)}
                    row = session.get(AgentToolCall, row.id) or row
                    row.status = "failed"
                    row.error_message = result["message"]
                    row.result_json = result
                except Exception as exc:  # noqa: BLE001
                    result = {"error": "tool_execution_failed", "message": untrusted_text(str(exc), limit=400)}
                    row = session.get(AgentToolCall, row.id) or row
                    row.status = "failed"
                    row.error_message = result["message"]
                    row.result_json = result
            else:
                row.status = "running"
                session.commit()
                try:
                    result = bound_json(spec.handler(session, arguments))
                    row = session.get(AgentToolCall, row.id) or row
                    run = self._get_run(session, run.id)
                    row.status = "completed"
                    row.result_json = result if isinstance(result, dict) else {"value": result}
                except DomainError as exc:
                    result = {"error": exc.error_code, "message": untrusted_text(exc.message, limit=400)}
                    row = session.get(AgentToolCall, row.id) or row
                    row.status = "failed"
                    row.error_message = result["message"]
                    row.result_json = result
                except Exception as exc:  # noqa: BLE001
                    result = {"error": "tool_execution_failed", "message": untrusted_text(str(exc), limit=400)}
                    row = session.get(AgentToolCall, row.id) or row
                    row.status = "failed"
                    row.error_message = result["message"]
                    row.result_json = result

            if not self._claim_running_step(session, run.id):
                row.status = "failed"
                row.error_message = "Cancelled by user."
                session.commit()
                return False
            envelope = {
                "tool_name": call.name,
                "tool_call_id": call.call_id or row.id,
                "result": row.result_json or {},
            }
            tool_message = AgentMessage(
                id=new_id("amsg"),
                session_id=run.session_id,
                run_id=run.id,
                role="tool",
                content=json.dumps(envelope, ensure_ascii=False, default=str),
                sequence=self._next_message_sequence(session, run.session_id),
            )
            session.add(tool_message)
            run = self._get_run(session, run.id)
            run.messages.append(tool_message)
            session.commit()
            if awaiting_approval:
                break
        return awaiting_approval

    def _execute_approved_payload(
        self,
        session: Session,
        approval: AgentApproval,
        *,
        reserved_task_id: str | None = None,
    ) -> tuple[str, str]:
        if approval.tool_name not in WRITE_TOOL_NAMES:
            raise ValidationError("unsupported_agent_write_tool", f"Unsupported write tool: {approval.tool_name}")
        payload = normalize_write_payload(approval.tool_name, approval.payload_json)
        task_id = reserved_task_id or approval.result_task_id
        action = approval.tool_name
        if action == "create_training_task":
            training = self._require_training_service()
            create_payload = TrainingTaskCreate.model_validate(
                {key: value for key, value in payload.items() if key not in {"action", "preview", "reserved_task_id"}}
            )
            task = training.create_task(session, create_payload, task_id=task_id)
            summary = (
                f"已确认并创建训练任务 `{task.id}`（数据集 {task.dataset_id}）。"
                f"可在训练页查看进度。本轮结束，不会自动等待训练完成。"
            )
            return task.id, summary
        if action == "create_model_evaluation":
            models = self._require_model_service()
            runner = self._require_evaluation_runner()
            model_id = str(payload.get("model_id") or "")
            split = str(payload.get("split") or "val")
            confidence = float(payload.get("confidence", 0.25))
            iou = float(payload.get("iou", 0.5))
            record = models.create_evaluation(
                session,
                model_id,
                split,
                confidence,
                iou,
                evaluation_id=task_id,
            )
            if record.status == "pending":
                runner.start_task(record.id)
            summary = (
                f"已确认并创建模型评估任务 `{record.id}`（模型 {model_id}，split={split}）。"
                f"可在模型详情页查看结果。本轮结束，不会自动串联下一步。"
            )
            return record.id, summary
        if action == "create_auto_annotation_task":
            auto = self._require_auto_annotation_service()
            dataset_id = str(payload.get("dataset_id") or "")
            request = AutoAnnotationCreateRequest.model_validate(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"action", "preview", "dataset_id", "reserved_task_id"}
                }
            )
            task = auto.create_task(session, dataset_id, request, task_id=task_id)
            summary = (
                f"已确认并创建自动标注任务 `{task.id}`（数据集 {dataset_id}，模型 {task.model_id}）。"
                f"结果须人工审核。本轮结束，不会自动串联下一步。"
            )
            return task.id, summary
        raise ValidationError("unsupported_agent_write_tool", f"Unsupported write tool: {action}")

    def _managed_task_exists(self, session: Session, tool_name: str, task_id: str) -> bool:
        if tool_name == "create_training_task":
            return session.get(TrainingTask, task_id) is not None
        if tool_name == "create_model_evaluation":
            return session.get(ModelEvaluationRecord, task_id) is not None
        if tool_name == "create_auto_annotation_task":
            return session.get(AutoAnnotationTask, task_id) is not None
        return False

    def _require_training_service(self) -> TrainingService:
        if self.training_service is None:
            if self.storage is None:
                raise ValidationError("agent_write_unavailable", "Training service is not configured.")
            from app.training.runtime.queue import training_queue

            self.training_service = TrainingService(self.session_factory, self.storage, training_queue)
        return self.training_service

    def _require_model_service(self) -> ModelService:
        if self.model_service is None:
            if self.storage is None:
                raise ValidationError("agent_write_unavailable", "Model service is not configured.")
            self.model_service = ModelService(self.storage)
        return self.model_service

    def _require_auto_annotation_service(self) -> AutoAnnotationService:
        if self.auto_annotation_service is None:
            if self.storage is None:
                raise ValidationError("agent_write_unavailable", "Auto-annotation service is not configured.")
            from app.auto_annotation.queue import auto_annotation_queue

            self.auto_annotation_service = AutoAnnotationService(self.session_factory, self.storage, auto_annotation_queue)
        return self.auto_annotation_service

    def _require_evaluation_runner(self) -> YoloEvaluationRunner:
        if self.evaluation_runner is None:
            if self.storage is None:
                raise ValidationError("agent_write_unavailable", "Evaluation runner is not configured.")
            self.evaluation_runner = YoloEvaluationRunner(self.session_factory, self.storage)
        return self.evaluation_runner

    def _get_approval(self, session: Session, approval_id: str) -> AgentApproval:
        approval = session.get(AgentApproval, approval_id)
        if approval is None:
            raise NotFoundError("agent_approval_not_found", "Agent approval was not found.")
        return approval

    def _provider_history(self, session: Session, session_id: str, run_id: str) -> list[ProviderMessage]:
        run = self._get_run(session, run_id)
        current_messages = sorted(run.messages, key=lambda item: item.sequence)
        first_sequence = min((item.sequence for item in current_messages), default=0)
        prior = list(
            session.scalars(
                select(AgentMessage)
                .where(AgentMessage.session_id == session_id, AgentMessage.sequence < first_sequence,
                       AgentMessage.role.in_(("user", "assistant")))
                .order_by(AgentMessage.sequence.desc(), AgentMessage.id.desc())
                .limit(24)
            )
        )
        history = list(reversed(prior)) + current_messages
        calls = list(session.scalars(
            select(AgentToolCall).where(AgentToolCall.run_id == run_id).order_by(AgentToolCall.sequence.asc())
        ))
        call_index = 0
        messages: list[ProviderMessage] = []
        for item in history:
            if item.role not in {"user", "assistant", "system", "tool"}:
                continue
            if item.role == "tool":
                if item.run_id != run_id or call_index >= len(calls):
                    continue
                try:
                    payload = json.loads(item.content)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                call = calls[call_index]
                call_index += 1
                call_id = str(payload.get("tool_call_id") or call.id)
                messages.append(ProviderMessage(
                    role="assistant", content="",
                    tool_calls=[ProviderToolCall(name=call.name, arguments=call.arguments_json or {}, call_id=call_id)],
                ))
                messages.append(ProviderMessage(
                    role="tool", content=item.content, name=call.name, tool_call_id=call_id,
                ))
                continue
            messages.append(ProviderMessage(role=item.role, content=item.content))
        instructions = [ProviderMessage(role="system", content=SYSTEM_PROMPT)]
        profile = get_profile(run.profile_id, run.profile_version)
        instructions.append(ProviderMessage(role="system", content=f"助手模式：{profile.title}（{profile.id}@{profile.version}）。{profile.instructions}"))
        instructions.append(ProviderMessage(role="system", content=(
            "本轮权限：" + ("只读，禁止写操作。" if run.read_only else "写操作必须经人工确认。")
        )))
        if run.context_json:
            instructions.append(ProviderMessage(role="system", content=(
                "The current question has these server-validated page context IDs: "
                + json.dumps(run.context_json, ensure_ascii=False)
                + ". Resolve 'this/current task/model/dataset' using this context. "
                "Explicit IDs in the question take precedence. Fetch facts through tools; "
                "context IDs alone are not evidence of status or quality."
            )))
        return instructions + bounded_history(messages)

    def _record_inference_step(
        self, session: Session, run_id: str, *, provider: str, model: str,
        source: str, duration_ms: int, outcome: str, reason: str | None,
    ) -> None:
        # Persist provenance independently of the answer, including failed or
        # cancelled calls. Only fixed reason codes enter SQLite, never raw errors.
        safe_reasons = {
            "agent_provider_http_error", "agent_provider_unreachable", "agent_provider_bad_response",
            "agent_provider_unsupported", "missing_evaluation_lookup", "no_tool_calls", "provider_error",
        }
        session.execute(update(AgentRun).where(AgentRun.id == run_id).values(updated_at=utc_now()))
        run = self._get_run(session, run_id)
        steps = list(run.inference_steps_json or [])
        steps.append({
            "round": len(steps) + 1,
            "source": source if source in {"llm", "mock", "fallback"} else "unknown",
            "provider": redact_text(provider, secrets=self._secret_values())[:64],
            "model": redact_text(model, secrets=self._secret_values())[:255],
            "duration_ms": max(0, duration_ms),
            "outcome": "discarded" if run.stop_requested else outcome,
            "reason": reason if reason in safe_reasons else ("provider_error" if reason else None),
        })
        run.inference_steps_json = steps
        session.commit()

    def _claim_running_step(self, session: Session, run_id: str) -> bool:
        """Serialize a step's finalization with cancellation, without holding the lock during I/O."""
        claimed = session.execute(update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status == "running", AgentRun.stop_requested.is_(False),
        ).values(updated_at=utc_now()))
        return claimed.rowcount == 1

    def _finish_run(self, session: Session, run: AgentRun, *, status: str, error: str | None = None) -> AgentRunResponse:
        run_id = run.id
        if status == "cancelled":
            session.rollback()
            return self.get_run(session, run_id)
        values: dict[str, Any] = {"status": status, "finished_at": utc_now()}
        if error:
            values["error_message"] = redact_text(error, secrets=self._secret_values())[:2000]
        finished = session.execute(update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status == "running", AgentRun.stop_requested.is_(False),
        ).values(**values))
        if finished.rowcount != 1:
            session.rollback()
            return self.get_run(session, run_id)
        session.commit()
        return self.get_run(session, run_id)

    def _session_summary(self, session: Session, row: AgentSession) -> AgentSessionResponse:
        message_count = session.scalar(
            select(func.count()).select_from(AgentMessage).where(AgentMessage.session_id == row.id)
        ) or 0
        latest_run = session.scalar(
            select(AgentRun)
            .where(AgentRun.session_id == row.id)
            .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
            .limit(1)
        )
        return AgentSessionResponse(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
            message_count=int(message_count),
            profile_id=row.profile_id, profile_version=row.profile_version,
            context=AgentContext.model_validate(row.context_json or {}),
            latest_run_status=None if latest_run is None else latest_run.status,  # type: ignore[arg-type]
        )

    def _get_session(self, session: Session, session_id: str) -> AgentSession:
        row = session.get(AgentSession, session_id)
        if row is None:
            raise NotFoundError("agent_session_not_found", "Agent session was not found.")
        return row

    def _get_run(self, session: Session, run_id: str) -> AgentRun:
        run = session.scalar(
            select(AgentRun)
            .where(AgentRun.id == run_id)
            .options(
                selectinload(AgentRun.messages),
                selectinload(AgentRun.tool_calls),
                selectinload(AgentRun.approvals),
            )
            .execution_options(populate_existing=True)
        )
        if run is None:
            raise NotFoundError("agent_run_not_found", "Agent run was not found.")
        return run

    def _next_message_sequence(self, session: Session, session_id: str) -> int:
        current = session.scalar(select(func.max(AgentMessage.sequence)).where(AgentMessage.session_id == session_id))
        return int(current or 0) + 1

    def create_approval_record(
        self,
        session: Session,
        *,
        run: AgentRun,
        tool_name: str,
        payload_json: dict,
        tool_call_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentApproval:
        """Helper for Week 4 write tools; Week 1/2 expose persistence shape via tests."""
        key = idempotency_key or new_id("aidem")
        existing = session.scalar(select(AgentApproval).where(AgentApproval.idempotency_key == key))
        if existing is not None:
            return existing
        row = AgentApproval(
            id=new_id("aappr"),
            run_id=run.id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            payload_json=payload_json,
            status="pending",
            idempotency_key=key,
            expires_at=utc_now() + timedelta(seconds=self.settings.agent_approval_ttl_seconds),
        )
        session.add(row)
        session.flush()
        return row
