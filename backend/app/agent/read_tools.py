from __future__ import annotations

from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.bounds import (
    MAX_ERROR_SAMPLES,
    MAX_ISSUES,
    MAX_LIST_ITEMS,
    MAX_LOG_TAIL_LINES,
    bound_json,
    bound_log_text,
    omit_paths,
    truncate_list,
    untrusted_text,
)
from app.agent.tools import AgentToolRegistry, AgentToolSpec
from app.core.models import Annotation
from app.core.storage import Storage
from app.dataset import service as dataset_service
from app.dataset.images import split_counts
from app.dataset.quality.service import DatasetQualityService
from app.dataset.validation import validate_dataset
from app.models.service import ModelService
from app.training.runtime.queue import training_queue
from app.training.service import TrainingService


ToolHandler = Callable[[Session, dict[str, Any]], dict[str, Any]]


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required argument: {key}")
    return value.strip()


def _require_str_any(arguments: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Missing required argument: one of {', '.join(keys)}")


def _optional_str(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid argument type: {key}")
    return value.strip()


def _tail(arguments: dict[str, Any], default: int = 40) -> int:
    raw = arguments.get("tail", default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("tail must be an integer") from exc
    return max(1, min(value, MAX_LOG_TAIL_LINES))


def _annotated_counts(session: Session, dataset_ids: list[str]) -> dict[str, int]:
    if not dataset_ids:
        return {}
    rows = session.execute(
        select(Annotation.dataset_id, func.count(func.distinct(Annotation.image_id)))
        .where(Annotation.dataset_id.in_(dataset_ids))
        .group_by(Annotation.dataset_id)
    ).all()
    return {dataset_id: count for dataset_id, count in rows}


def _dataset_card(session: Session, dataset) -> dict[str, Any]:
    annotated = _annotated_counts(session, [dataset.id]).get(dataset.id, 0)
    return {
        "id": dataset.id,
        "name": untrusted_text(dataset.name),
        "description": untrusted_text(dataset.description, limit=240) if dataset.description else None,
        "task_type": dataset.task_type,
        "image_count": dataset.image_count,
        "annotated_image_count": annotated,
        "class_count": dataset.class_count,
    }


def build_read_handlers(
    storage: Storage,
    *,
    training_service: TrainingService | None = None,
    model_service: ModelService | None = None,
    session_factory=None,
) -> dict[str, tuple[str, dict[str, Any], ToolHandler]]:
    """Return name -> (description, json_schema, handler)."""
    training = training_service or TrainingService(session_factory, storage, training_queue)
    models = model_service or ModelService(storage)
    quality = DatasetQualityService()

    def list_datasets(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        rows = dataset_service.list_datasets(session)
        items, truncated = truncate_list([_dataset_card(session, row) for row in rows])
        return bound_json({"items": items, "total": len(rows), "truncated": truncated})

    def get_dataset(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = _require_str(arguments, "dataset_id")
        dataset = dataset_service.get_dataset(session, dataset_id)
        classes = [
            {
                "id": item.id,
                "class_index": item.class_index,
                "name": untrusted_text(item.name),
            }
            for item in dataset_service.list_classes(session, dataset_id)
        ]
        classes, class_truncated = truncate_list(classes)
        card = _dataset_card(session, dataset)
        listed = training.list_tasks(session, dataset_id=dataset_id)
        latest_training = listed.items[0] if listed.items else None
        card.update(
            {
                "classes": classes,
                "classes_truncated": class_truncated,
                "split_counts": dict(split_counts(session, dataset_id)),
                # Upstream dataset.summary fields (bounded, no paths).
                "latest_training": (
                    {"id": latest_training.id, "status": latest_training.status} if latest_training else None
                ),
            }
        )
        return bound_json(card)

    def dataset_quality_report(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = _require_str(arguments, "dataset_id")
        report = quality.report(session, dataset_id)
        payload = report.model_dump()
        payload["class_distribution"] = [
            {
                **item,
                "name": untrusted_text(item.get("name")),
            }
            for item in payload.get("class_distribution") or []
        ]
        issues = [
            {
                **issue,
                "message": untrusted_text(issue.get("message"), limit=240),
                "annotation_ids": (issue.get("annotation_ids") or [])[:10],
            }
            for issue in payload.get("issues") or []
        ]
        issues, issues_truncated = truncate_list(issues, limit=MAX_ISSUES)
        payload["issues"] = issues
        payload["issues_truncated"] = issues_truncated
        return bound_json(payload)

    def validate_dataset_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = _require_str(arguments, "dataset_id")
        report = validate_dataset(session, storage, dataset_id)
        payload = report.model_dump()
        issues = [
            {
                **issue,
                "message": untrusted_text(issue.get("message"), limit=240),
            }
            for issue in payload.get("issues") or []
        ]
        issues, issues_truncated = truncate_list(issues, limit=MAX_ISSUES)
        payload["issues"] = issues
        payload["issues_truncated"] = issues_truncated
        return bound_json(payload)

    def list_training_tasks(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = _optional_str(arguments, "dataset_id")
        listed = training.list_tasks(session, dataset_id=dataset_id)
        items = []
        for task in listed.items:
            items.append(
                {
                    "id": task.id,
                    "dataset_id": task.dataset_id,
                    "name": untrusted_text(task.name),
                    "status": task.status,
                    "task_type": task.task_type,
                    "model_name": untrusted_text(task.model_name),
                    "progress_epoch": task.progress_epoch,
                    "progress_total_epochs": task.progress_total_epochs,
                    "progress_percent": task.progress_percent,
                    "error_message": untrusted_text(task.error_message, limit=240) if task.error_message else None,
                }
            )
        items, truncated = truncate_list(items)
        return bound_json({"items": items, "total": len(listed.items), "truncated": truncated})

    def get_training_task(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = _require_str_any(arguments, "task_id", "training_task_id")
        task = training.get_task_response(session, task_id)
        payload = omit_paths(
            task.model_dump(mode="json"),
            "model_path",
            "export_path",
            "data_yaml_path",
            "run_dir",
            "logs_path",
            "summary_path",
            "best_model_path",
            "last_model_path",
            "command_preview",
        )
        payload["name"] = untrusted_text(payload.get("name"))
        payload["model_name"] = untrusted_text(payload.get("model_name"))
        if payload.get("error_message"):
            payload["error_message"] = untrusted_text(payload.get("error_message"), limit=240)
        payload["metrics"] = payload.pop("metrics_json", {}) or {}
        return bound_json(payload)

    def training_summary_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = _require_str_any(arguments, "task_id", "training_task_id")
        summary = training.summary(session, task_id)
        payload = summary.model_dump(mode="json")
        log_summary = dict(payload.get("log_summary") or {})
        if "tail" in log_summary:
            log_summary["tail"] = bound_log_text(log_summary.get("tail"))
        payload["log_summary"] = log_summary
        payload["risks"] = [untrusted_text(item, limit=240) for item in (payload.get("risks") or [])[:10]]
        payload["next_steps"] = [untrusted_text(item, limit=240) for item in (payload.get("next_steps") or [])[:10]]
        dataset = dict(payload.get("dataset") or {})
        if "name" in dataset:
            dataset["name"] = untrusted_text(dataset.get("name"))
        payload["dataset"] = dataset
        return bound_json(payload)

    def training_logs_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = _require_str_any(arguments, "task_id", "training_task_id")
        tail = _tail(arguments)
        logs = training.logs(session, task_id, tail=tail)
        preview = bound_log_text(logs.logs)
        return bound_json(
            {
                "task_id": logs.task_id,
                "line_count": logs.line_count,
                "tail": tail,
                "preview": preview,
            }
        )

    def list_models_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset_id = _optional_str(arguments, "dataset_id")
        include_archived = bool(arguments.get("include_archived", False))
        listed = models.list_models(session, dataset_id=dataset_id, include_archived=include_archived)
        items = []
        for model in listed.items:
            items.append(
                {
                    "id": model.id,
                    "name": untrusted_text(model.name),
                    "dataset_id": model.dataset_id,
                    "training_task_id": model.training_task_id,
                    "task_type": model.task_type,
                    "artifact_type": model.artifact_type,
                    "format": model.format,
                    "status": model.status,
                    "precision": model.precision,
                    "recall": model.recall,
                    "map50": model.map50,
                    "map50_95": model.map50_95,
                    "notes": untrusted_text(model.notes, limit=160),
                }
            )
        items, truncated = truncate_list(items)
        return bound_json({"items": items, "total": listed.total, "truncated": truncated})

    def get_model_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = _require_str_any(arguments, "model_id", "model_version_id")
        model = models.get_model_response(session, model_id)
        payload = omit_paths(model.model_dump(mode="json"), "model_path")
        payload["name"] = untrusted_text(payload.get("name"))
        payload["notes"] = untrusted_text(payload.get("notes"), limit=240)
        payload["base_model"] = untrusted_text(payload.get("base_model")) if payload.get("base_model") else None
        return bound_json(payload)

    def compare_models_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        baseline_id = _require_str_any(arguments, "baseline_model_id", "baseline_model_version_id")
        candidate_id = _require_str_any(arguments, "candidate_model_id", "candidate_model_version_id")
        result = models.compare(session, baseline_id, candidate_id)
        for side in ("baseline", "candidate"):
            block = dict(result.get(side) or {})
            block["name"] = untrusted_text(block.get("name"))
            result[side] = block
        result["suggestions"] = [untrusted_text(item, limit=240) for item in (result.get("suggestions") or [])[:8]]
        return bound_json(result)

    def list_model_evaluations_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = _require_str_any(arguments, "model_id", "model_version_id")
        records = models.list_evaluations(session, model_id)
        items = []
        for record in records:
            metrics = (record.result_json or {}).get("metrics") or {}
            items.append(
                {
                    "id": record.id,
                    "model_id": record.model_id,
                    "dataset_id": record.dataset_id,
                    "split": record.split,
                    "status": record.status,
                    "confidence": record.confidence,
                    "iou": record.iou,
                    "metrics": metrics,
                    "error_message": untrusted_text(record.error_message, limit=240) if record.error_message else None,
                }
            )
        items, truncated = truncate_list(items)
        return bound_json({"items": items, "total": len(records), "truncated": truncated})

    def get_model_evaluation_tool(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = _require_str_any(arguments, "model_id", "model_version_id")
        evaluation_id = _optional_str(arguments, "evaluation_id") or _optional_str(arguments, "evaluation_task_id")
        if evaluation_id:
            record = models.get_evaluation(session, model_id, evaluation_id)
        else:
            records = models.list_evaluations(session, model_id)
            if not records:
                return bound_json({"found": False, "model_id": model_id})
            record = records[0]
        result_json = dict(record.result_json or {})
        error_samples = result_json.get("error_samples") or []
        if isinstance(error_samples, list):
            samples, truncated = truncate_list(error_samples, limit=MAX_ERROR_SAMPLES)
            # Drop bulky fields; keep type/confidence style summaries when present.
            compact_samples = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                compact_samples.append(
                    {
                        key: sample.get(key)
                        for key in ("type", "kind", "confidence", "image_id", "class_name", "message")
                        if key in sample
                    }
                )
            result_json["error_samples"] = compact_samples
            result_json["error_samples_truncated"] = truncated
        result_json.pop("artifacts", None)
        return bound_json(
            {
                "id": record.id,
                "model_id": record.model_id,
                "dataset_id": record.dataset_id,
                "split": record.split,
                "status": record.status,
                "confidence": record.confidence,
                "iou": record.iou,
                "metrics": result_json.get("metrics") or {},
                "error_samples": result_json.get("error_samples") or [],
                "error_samples_truncated": result_json.get("error_samples_truncated", 0),
                "error_message": untrusted_text(record.error_message, limit=240) if record.error_message else None,
            }
        )

    def model_latest_for_dataset(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        """Upstream model.latest_for_dataset — latest active managed PT for a dataset."""
        dataset_id = _optional_str(arguments, "dataset_id")
        listed = models.list_models(session, dataset_id=dataset_id, include_archived=False)
        for model in listed.items:
            if model.format != "pt":
                continue
            return bound_json(
                {
                    "found": True,
                    "model": {
                        "id": model.id,
                        "name": untrusted_text(model.name),
                        "dataset_id": model.dataset_id,
                        "task_type": model.task_type,
                        "artifact_type": model.artifact_type,
                        "format": model.format,
                        "status": model.status,
                        "map50": model.map50,
                        "map50_95": model.map50_95,
                    },
                }
            )
        return bound_json({"found": False})

    def evaluation_error_summary(session: Session, arguments: dict[str, Any]) -> dict[str, Any]:
        """Upstream evaluation.error_summary — compact error-sample histogram."""
        model_id = _require_str_any(arguments, "model_id", "model_version_id")
        evaluation_id = _optional_str(arguments, "evaluation_id") or _optional_str(arguments, "evaluation_task_id")
        if evaluation_id:
            record = models.get_evaluation(session, model_id, evaluation_id)
        else:
            records = models.list_evaluations(session, model_id)
            record = records[0] if records else None
            if record is None:
                return bound_json({"found": False})
        samples = (record.result_json or {}).get("error_samples") or []
        counts: dict[str, int] = {}
        if isinstance(samples, list):
            for sample in samples[:100]:
                if not isinstance(sample, dict):
                    continue
                kind = str(sample.get("type") or sample.get("kind") or "unknown")
                counts[kind] = counts.get(kind, 0) + 1
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
        return bound_json(
            {
                "found": True,
                "evaluation_id": record.id,
                "model_id": record.model_id,
                "dataset_id": record.dataset_id,
                "status": record.status,
                "error_type_counts": [{"type": key, "count": value} for key, value in top],
                "sample_count": len(samples) if isinstance(samples, list) else 0,
            }
        )

    string_id = {"type": "string", "minLength": 1, "maxLength": 64}
    # Primary names follow upstream domain.action semantics with OpenAI-safe underscores.
    # Legacy aliases keep Week 2–5 tests and old mock plans working.
    primary: dict[str, tuple[str, dict[str, Any], ToolHandler]] = {
        "global_summary": (
            "Summarize local workspace datasets (upstream global.summary).",
            {"type": "object", "properties": {}, "additionalProperties": False},
            list_datasets,
        ),
        "dataset_summary": (
            "Summarize one dataset: classes, splits, latest training (upstream dataset.summary).",
            {
                "type": "object",
                "properties": {"dataset_id": string_id},
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
            get_dataset,
        ),
        "dataset_quality_report": (
            "Read-only dataset quality report (upstream dataset.quality_report).",
            {
                "type": "object",
                "properties": {"dataset_id": string_id},
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
            dataset_quality_report,
        ),
        "dataset_validate": (
            "Validate dataset annotations and files (upstream dataset.validate).",
            {
                "type": "object",
                "properties": {"dataset_id": string_id},
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
            validate_dataset_tool,
        ),
        "training_list": (
            "List local training tasks (Starter list helper for training.status discovery).",
            {
                "type": "object",
                "properties": {"dataset_id": string_id},
                "additionalProperties": False,
            },
            list_training_tasks,
        ),
        "training_status": (
            "Get one training task status and metrics (upstream training.status).",
            {
                "type": "object",
                "properties": {
                    "task_id": string_id,
                    "training_task_id": string_id,
                },
                "additionalProperties": False,
            },
            get_training_task,
        ),
        "training_latest_result": (
            "Bounded training summary / latest result (upstream training.latest_result).",
            {
                "type": "object",
                "properties": {
                    "task_id": string_id,
                    "training_task_id": string_id,
                },
                "additionalProperties": False,
            },
            training_summary_tool,
        ),
        "training_logs": (
            "Short training log tail (upstream training.logs).",
            {
                "type": "object",
                "properties": {
                    "task_id": string_id,
                    "training_task_id": string_id,
                    "tail": {"type": "integer", "minimum": 1, "maximum": MAX_LOG_TAIL_LINES},
                },
                "additionalProperties": False,
            },
            training_logs_tool,
        ),
        "model_list": (
            "List managed model versions.",
            {
                "type": "object",
                "properties": {
                    "dataset_id": string_id,
                    "include_archived": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            list_models_tool,
        ),
        "model_latest_for_dataset": (
            "Latest active managed PT for a dataset (upstream model.latest_for_dataset).",
            {
                "type": "object",
                "properties": {"dataset_id": string_id},
                "additionalProperties": False,
            },
            model_latest_for_dataset,
        ),
        "model_get": (
            "Get one managed model card without filesystem paths.",
            {
                "type": "object",
                "properties": {
                    "model_id": string_id,
                    "model_version_id": string_id,
                },
                "additionalProperties": False,
            },
            get_model_tool,
        ),
        "model_compare": (
            "Compare two models from the same dataset (upstream model.compare).",
            {
                "type": "object",
                "properties": {
                    "baseline_model_id": string_id,
                    "candidate_model_id": string_id,
                    "baseline_model_version_id": string_id,
                    "candidate_model_version_id": string_id,
                    "dataset_id": string_id,
                },
                "additionalProperties": False,
            },
            compare_models_tool,
        ),
        "evaluation_list": (
            "List evaluations for a managed model.",
            {
                "type": "object",
                "properties": {
                    "model_id": string_id,
                    "model_version_id": string_id,
                },
                "additionalProperties": False,
            },
            list_model_evaluations_tool,
        ),
        "evaluation_summary": (
            "Evaluation summary / detail (upstream evaluation.summary).",
            {
                "type": "object",
                "properties": {
                    "model_id": string_id,
                    "model_version_id": string_id,
                    "evaluation_id": string_id,
                    "evaluation_task_id": string_id,
                },
                "additionalProperties": False,
            },
            get_model_evaluation_tool,
        ),
        "evaluation_error_summary": (
            "Compact evaluation error-sample counts (upstream evaluation.error_summary).",
            {
                "type": "object",
                "properties": {
                    "model_id": string_id,
                    "model_version_id": string_id,
                    "evaluation_id": string_id,
                    "evaluation_task_id": string_id,
                },
                "additionalProperties": False,
            },
            evaluation_error_summary,
        ),
    }

    legacy_aliases = {
        "list_datasets": "global_summary",
        "get_dataset": "dataset_summary",
        "validate_dataset": "dataset_validate",
        "list_training_tasks": "training_list",
        "get_training_task": "training_status",
        "training_summary": "training_latest_result",
        "list_models": "model_list",
        "get_model": "model_get",
        "compare_models": "model_compare",
        "list_model_evaluations": "evaluation_list",
        "get_model_evaluation": "evaluation_summary",
    }
    for alias, target in legacy_aliases.items():
        description, parameters, handler = primary[target]
        primary[alias] = (f"{description} Legacy alias for {target}.", parameters, handler)
    primary["__legacy_aliases__"] = legacy_aliases  # type: ignore[assignment]
    return primary


def build_default_registry(
    storage: Storage | None = None,
    *,
    training_service: TrainingService | None = None,
    model_service: ModelService | None = None,
    session_factory=None,
) -> AgentToolRegistry:
    registry = AgentToolRegistry()
    if storage is None:
        return registry
    handlers = build_read_handlers(
        storage,
        training_service=training_service,
        model_service=model_service,
        session_factory=session_factory,
    )
    legacy_aliases = handlers.pop("__legacy_aliases__", {})  # type: ignore[assignment]
    if not isinstance(legacy_aliases, dict):
        legacy_aliases = {}
    for name, (description, parameters, handler) in handlers.items():
        registry.register(
            AgentToolSpec(
                name=name,
                kind="read",
                description=description,
                parameters=parameters,
                handler=handler,
                expose_to_provider=name not in legacy_aliases,
            )
        )
    from app.agent.write_tools import register_write_tools

    register_write_tools(
        registry,
        storage,
        training_service=training_service,
        model_service=model_service,
        session_factory=session_factory,
    )
    return registry
