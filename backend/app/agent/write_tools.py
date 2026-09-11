from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.bounds import bound_json, untrusted_text
from app.agent.tools import AgentToolRegistry, AgentToolSpec
from app.auto_annotation.schemas import AutoAnnotationCreateRequest
from app.core.errors import NotFoundError, ValidationError
from app.core.models import ImageItem, ModelVersion
from app.core.storage import Storage
from app.dataset.service import get_dataset
from app.models.schemas import ModelEvaluationRequest
from app.models.service import ModelService
from app.training.config import validate_model_family
from app.training.schemas import TrainingTaskCreate
from app.training.service import TrainingService
from app.core.task_types import TaskType


ToolHandler = Callable[[Session, dict[str, Any]], dict[str, Any]]

WRITE_TOOL_NAMES = frozenset(
    {"create_training_task", "create_model_evaluation", "create_auto_annotation_task"}
)


def _pydantic_error(exc: PydanticValidationError) -> ValidationError:
    first = exc.errors()[0] if exc.errors() else {"msg": "invalid payload", "loc": ()}
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = untrusted_text(first.get("msg") or "invalid payload", limit=240)
    if location:
        message = f"{location}: {message}"
    return ValidationError("invalid_agent_write_payload", message)


def normalize_write_payload(tool_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Re-validate approval payloads at confirm time; drop untrusted extra keys."""
    raw = dict(payload or {})
    if tool_name not in WRITE_TOOL_NAMES:
        raise ValidationError("unsupported_agent_write_tool", f"Unsupported write tool: {tool_name}")
    if tool_name == "create_training_task":
        try:
            parsed = TrainingTaskCreate.model_validate(
                {key: value for key, value in raw.items() if key not in {"action", "preview", "reserved_task_id"}}
            )
        except PydanticValidationError as exc:
            raise _pydantic_error(exc) from exc
        data = parsed.model_dump(mode="json")
        data["action"] = "create_training_task"
        data["preview"] = {
            "dataset_id": parsed.dataset_id,
            "name": untrusted_text(parsed.name),
            "model": untrusted_text(parsed.model),
            "epochs": parsed.epochs,
            "img_size": parsed.img_size,
            "batch_size": parsed.batch_size,
            "device": parsed.device,
            "val_ratio": parsed.val_ratio,
            "seed": parsed.seed,
        }
        return data
    if tool_name == "create_model_evaluation":
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValidationError("invalid_agent_write_payload", "model_id is required.")
        try:
            request = ModelEvaluationRequest.model_validate(
                {key: value for key, value in raw.items() if key not in {"action", "preview", "model_id", "reserved_task_id"}}
            )
        except PydanticValidationError as exc:
            raise _pydantic_error(exc) from exc
        data = {"action": "create_model_evaluation", "model_id": model_id.strip(), **request.model_dump(mode="json")}
        data["preview"] = {
            "model_id": data["model_id"],
            "split": data["split"],
            "confidence": data["confidence"],
            "iou": data["iou"],
        }
        return data
    dataset_id = raw.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValidationError("invalid_agent_write_payload", "dataset_id is required.")
    try:
        request = AutoAnnotationCreateRequest.model_validate(
            {key: value for key, value in raw.items() if key not in {"action", "preview", "dataset_id", "reserved_task_id"}}
        )
    except PydanticValidationError as exc:
        raise _pydantic_error(exc) from exc
    data = {"action": "create_auto_annotation_task", "dataset_id": dataset_id.strip(), **request.model_dump(mode="json")}
    data["preview"] = {
        "dataset_id": data["dataset_id"],
        "model_id": data["model_id"],
        "confidence": data["confidence"],
        "iou": data["iou"],
        "skip_annotated_images": data["skip_annotated_images"],
        "clean_old_annotations": data["clean_old_annotations"],
    }
    return data


def _preflight_training(session: Session, storage: Storage, payload: TrainingTaskCreate, training: TrainingService) -> None:
    dataset = get_dataset(session, payload.dataset_id)
    task_type = payload.task_type or TaskType(dataset.task_type)
    if task_type.value != dataset.task_type:
        raise ValidationError("task_type_mismatch", "Training task type must match the dataset task type.")
    validate_model_family(task_type, payload.model)
    training._validate_dataset_ready(session, dataset)


def _preflight_evaluation(session: Session, models: ModelService, model_id: str, split: str) -> None:
    model = models.get_model(session, model_id)
    if not model.dataset_id:
        raise ValidationError("evaluation_dataset_missing", "This managed model is not attached to a dataset.")
    if model.format != "pt" or model.engine_type != "ultralytics":
        raise ValidationError("unsupported_model_engine", "Local evaluation requires a managed Ultralytics PT model.")
    image_count = session.scalar(
        select(func.count()).select_from(ImageItem).where(ImageItem.dataset_id == model.dataset_id, ImageItem.split == split)
    ) or 0
    if not image_count:
        raise ValidationError("evaluation_split_empty", f"The {split} split has no images to evaluate.")


def _preflight_auto_annotation(session: Session, storage: Storage, dataset_id: str, request: AutoAnnotationCreateRequest) -> None:
    dataset = get_dataset(session, dataset_id)
    model = session.get(ModelVersion, request.model_id)
    if model is None:
        raise NotFoundError("model_not_found", "Model version was not found.")
    if model.status != "active":
        raise ValidationError("model_archived", "Auto annotation requires an active managed model.")
    if model.format != "pt" or model.engine_type != "ultralytics":
        raise ValidationError("unsupported_model_engine", "Auto annotation requires a managed Ultralytics PT model.")
    if model.task_type != dataset.task_type:
        raise ValidationError("auto_annotation_task_mismatch", "The model task type must match the dataset task type.")
    model_path = storage.managed_model_path(model.model_path)
    if not model_path.is_file():
        raise NotFoundError("model_file_missing", "The managed model file is missing.")


def preflight_write_payload(
    session: Session,
    storage: Storage,
    tool_name: str,
    payload: dict[str, Any],
    *,
    training_service: TrainingService | None = None,
    model_service: ModelService | None = None,
    session_factory=None,
) -> None:
    """Cheap domain checks after normalize — used at propose and confirm-with-override."""
    from app.training.runtime.queue import training_queue

    training = training_service or TrainingService(session_factory, storage, training_queue)
    models = model_service or ModelService(storage)
    if tool_name == "create_training_task":
        create_payload = TrainingTaskCreate.model_validate(
            {key: value for key, value in payload.items() if key not in {"action", "preview", "reserved_task_id"}}
        )
        _preflight_training(session, storage, create_payload, training)
        return
    if tool_name == "create_model_evaluation":
        _preflight_evaluation(session, models, str(payload["model_id"]), str(payload.get("split") or "val"))
        return
    if tool_name == "create_auto_annotation_task":
        request = AutoAnnotationCreateRequest.model_validate(
            {
                key: value
                for key, value in payload.items()
                if key not in {"action", "preview", "dataset_id", "reserved_task_id"}
            }
        )
        _preflight_auto_annotation(session, storage, str(payload["dataset_id"]), request)
        return
    raise ValidationError("unsupported_agent_write_tool", f"Unsupported write tool: {tool_name}")


def build_write_handlers(
    storage: Storage,
    *,
    training_service: TrainingService | None = None,
    model_service: ModelService | None = None,
    session_factory=None,
) -> dict[str, tuple[str, dict[str, Any], ToolHandler]]:
    """Write tools validate/normalize payloads and run cheap preflight; execution waits for approval."""
    from app.training.runtime.queue import training_queue

    training = training_service or TrainingService(session_factory, storage, training_queue)
    models = model_service or ModelService(storage)

    string_id = {"type": "string", "minLength": 1, "maxLength": 64}

    def create_training_task(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        data = normalize_write_payload("create_training_task", arguments)
        payload = TrainingTaskCreate.model_validate(
            {key: value for key, value in data.items() if key not in {"action", "preview"}}
        )
        _preflight_training(session, storage, payload, training)
        return bound_json(data)

    def create_model_evaluation(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        data = normalize_write_payload("create_model_evaluation", arguments)
        _preflight_evaluation(session, models, str(data["model_id"]), str(data["split"]))
        return bound_json(data)

    def create_auto_annotation_task(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        data = normalize_write_payload("create_auto_annotation_task", arguments)
        request = AutoAnnotationCreateRequest.model_validate(
            {key: value for key, value in data.items() if key not in {"action", "preview", "dataset_id"}}
        )
        _preflight_auto_annotation(session, storage, str(data["dataset_id"]), request)
        return bound_json(data)

    return {
        "create_training_task": (
            "Propose creating a local training task. Requires human approval before execution.",
            {
                "type": "object",
                "properties": {
                    "dataset_id": string_id,
                    "name": {"type": "string", "minLength": 1, "maxLength": 255},
                    "model": {"type": "string", "minLength": 1, "maxLength": 255},
                    "task_type": {"type": "string", "enum": ["detect", "segment", "obb", "classify"]},
                    "epochs": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "img_size": {"type": "integer", "minimum": 32, "maximum": 4096},
                    "batch_size": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "device": {"type": "string", "minLength": 1, "maxLength": 64},
                    "workers": {"type": "integer", "minimum": 0, "maximum": 128},
                    "val_ratio": {"type": "number", "minimum": 0, "maximum": 0.9},
                    "seed": {"type": "integer", "minimum": 0},
                    "optimizer": {"type": "string", "maxLength": 64},
                    "lr0": {"type": "number", "exclusiveMinimum": 0},
                    "patience": {"type": "integer", "minimum": 0},
                },
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
            create_training_task,
        ),
        "create_model_evaluation": (
            "Propose evaluating a managed PT model on a persisted split. Requires human approval.",
            {
                "type": "object",
                "properties": {
                    "model_id": string_id,
                    "split": {"type": "string", "enum": ["train", "val", "test"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "iou": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["model_id"],
                "additionalProperties": False,
            },
            create_model_evaluation,
        ),
        "create_auto_annotation_task": (
            "Propose a dataset auto-annotation task with a managed PT. Requires human approval.",
            {
                "type": "object",
                "properties": {
                    "dataset_id": string_id,
                    "model_id": string_id,
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "iou": {"type": "number", "minimum": 0, "maximum": 1},
                    "clean_old_annotations": {"type": "boolean"},
                    "skip_annotated_images": {"type": "boolean"},
                    "class_mapping": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["dataset_id", "model_id"],
                "additionalProperties": False,
            },
            create_auto_annotation_task,
        ),
    }


def register_write_tools(
    registry: AgentToolRegistry,
    storage: Storage,
    *,
    training_service: TrainingService | None = None,
    model_service: ModelService | None = None,
    session_factory=None,
) -> None:
    for name, (description, parameters, handler) in build_write_handlers(
        storage,
        training_service=training_service,
        model_service=model_service,
        session_factory=session_factory,
    ).items():
        registry.register(
            AgentToolSpec(
                name=name,
                kind="write",
                description=description,
                parameters=parameters,
                handler=handler,
            )
        )


def reserved_id_prefix(tool_name: str) -> str:
    if tool_name == "create_training_task":
        return "train"
    if tool_name == "create_model_evaluation":
        return "eval"
    if tool_name == "create_auto_annotation_task":
        return "auto"
    raise ValidationError("unsupported_agent_write_tool", f"Unsupported write tool: {tool_name}")
