from __future__ import annotations

import math
from pathlib import Path
from typing import BinaryIO

import cv2
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Dataset, VideoImportTask
from app.core.storage import Storage
from app.training.observability.log_store import TrainingLogStore
from app.video_import.schemas import SamplingMode, VideoImportTaskResponse


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}
VALID_TASK_TYPES = {"detect", "segment", "obb", "classify"}
VALID_SPLITS = {"train", "val", "test"}
VALID_SPLIT_STRATEGIES = {"single", "time_blocks"}


def task_response(task: VideoImportTask) -> VideoImportTaskResponse:
    return VideoImportTaskResponse.model_validate(task)


class VideoImportService:
    def __init__(self, session_factory: sessionmaker[Session], storage: Storage, settings: Settings, queue) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.settings = settings
        self.queue = queue

    def create_task(
        self,
        session: Session,
        *,
        source: BinaryIO,
        source_file_name: str,
        name: str,
        task_type: str,
        split: str,
        sampling_mode: SamplingMode,
        sampling_value: float,
        start_seconds: float,
        end_seconds: float | None,
        split_strategy: str,
        time_block_seconds: float,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> VideoImportTask:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValidationError("video_import_name_empty", "Dataset name cannot be empty.")
        if task_type not in VALID_TASK_TYPES:
            raise ValidationError("video_import_task_type_invalid", "Choose a supported dataset task type.")
        if split not in VALID_SPLITS:
            raise ValidationError("video_import_split_invalid", "Choose train, val, or test as the initial split.")
        suffix = Path(source_file_name).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValidationError("video_format_unsupported", "Video import supports MP4, MOV, and AVI files.")
        config = self._normalize_config(
            sampling_mode,
            sampling_value,
            start_seconds,
            end_seconds,
            split_strategy,
            time_block_seconds,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        config["split"] = split
        task_id = new_id("vid")
        source_storage_name = f"source{suffix}"
        try:
            source_path, size_bytes, checksum = self.storage.write_video_upload(
                task_id,
                source_storage_name,
                source,
                max_bytes=self.settings.max_video_upload_bytes,
            )
            video_info = inspect_video(source_path, source_file_name, size_bytes)
            self._validate_preflight(video_info, config)
            estimated_images = estimate_images(video_info, config)
            estimated_output_bytes, sampled_frames = estimate_output_bytes(source_path, video_info, config, estimated_images)
            video_info["estimated_output_bytes"] = estimated_output_bytes
            video_info["estimate_sampled_frames"] = sampled_frames
            video_info["estimated_split_counts"] = estimate_split_counts(video_info, config, estimated_images)
            if estimated_output_bytes > self.settings.max_video_output_bytes:
                raise ValidationError("video_output_estimate_exceeded", "Estimated generated frame data exceeds the configured output limit.")
            task = VideoImportTask(
                id=task_id,
                name=cleaned_name,
                task_type=task_type,
                split=split,
                status="pending",
                start_requested=False,
                config_json=config,
                source_file_name=Path(source_file_name).name or "video",
                source_storage_name=source_storage_name,
                source_checksum=checksum,
                video_info_json=video_info,
                total_images=estimated_images,
                logs_path=str(self.storage.video_import_task_dir(task_id) / "video_import.log"),
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            TrainingLogStore(task.logs_path or "").append(
                f"Preflight passed: {video_info['width']}x{video_info['height']}, "
                f"{video_info['duration_seconds']:.2f}s, estimated {estimated_images} frame(s)."
            )
            return task
        except Exception:
            session.rollback()
            self.storage.remove_video_import_task(task_id)
            raise

    def get_task(self, session: Session, task_id: str) -> VideoImportTask:
        task = session.get(VideoImportTask, task_id)
        if task is None:
            raise NotFoundError("video_import_task_not_found", "Video import task was not found.")
        return task

    def start_task(self, session: Session, task_id: str) -> VideoImportTask:
        task = self.get_task(session, task_id)
        if task.status == "running":
            return task
        if task.status == "completed":
            raise ConflictError("video_import_completed", "This video import has already completed.")
        if task.status == "failed":
            raise ConflictError("video_import_retry_required", "Retry the failed import instead of starting it again.")
        source_path = self.storage.video_import_source_path(task.id, task.source_storage_name)
        if not source_path.is_file():
            raise NotFoundError("video_import_source_missing", "The staged video source is missing; upload it again.")
        task.start_requested = True
        session.commit()
        session.refresh(task)
        self.queue.submit(task.id)
        return task

    def retry_task(self, session: Session, task_id: str) -> VideoImportTask:
        task = self.get_task(session, task_id)
        if task.status != "failed":
            raise ConflictError("video_import_not_failed", "Only a failed video import can be retried.")
        source_path = self.storage.video_import_source_path(task.id, task.source_storage_name)
        if not source_path.is_file():
            raise NotFoundError("video_import_source_missing", "The staged video source is missing; upload it again.")
        dataset_id = task.dataset_id
        task.dataset_id = None
        task.status = "pending"
        task.start_requested = True
        task.generated_images = 0
        task.progress_percent = 0.0
        task.checkpoint_next_frame_index = 0
        task.checkpoint_next_sample_at = None
        task.output_bytes = 0
        task.error_message = None
        if dataset_id:
            dataset = session.get(Dataset, dataset_id)
            if dataset is not None:
                session.delete(dataset)
        session.commit()
        if dataset_id:
            self.storage.remove_dataset(dataset_id)
        session.refresh(task)
        TrainingLogStore(task.logs_path or "").append("Retry requested; prior generated frames were removed.")
        self.queue.submit(task.id)
        return task

    @staticmethod
    def _normalize_config(
        mode: SamplingMode,
        value: float,
        start_seconds: float,
        end_seconds: float | None,
        split_strategy: str,
        time_block_seconds: float,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> dict:
        if mode not in {"fps", "interval_seconds", "frame_interval"}:
            raise ValidationError("video_sampling_mode_invalid", "Choose FPS, time interval, or frame interval sampling.")
        if not math.isfinite(value) or value <= 0:
            raise ValidationError("video_sampling_value_invalid", "Sampling value must be a finite number greater than zero.")
        if mode == "frame_interval" and not value.is_integer():
            raise ValidationError("video_frame_interval_invalid", "Frame interval must be a whole number.")
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValidationError("video_start_invalid", "Start time must be zero or greater.")
        if end_seconds is not None:
            if not math.isfinite(end_seconds) or end_seconds <= start_seconds:
                raise ValidationError("video_end_invalid", "End time must be greater than start time.")
        if split_strategy not in VALID_SPLIT_STRATEGIES:
            raise ValidationError("video_split_strategy_invalid", "Choose a single split or time-block split strategy.")
        ratios = (train_ratio, val_ratio, test_ratio)
        if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
            raise ValidationError("video_split_ratio_invalid", "Time-block split ratios must be non-negative and sum to 1.")
        if split_strategy == "time_blocks":
            if not math.isfinite(time_block_seconds) or time_block_seconds <= 0:
                raise ValidationError("video_time_block_invalid", "Time block size must be a finite number greater than zero.")
            if not any(ratio > 0 for ratio in ratios):
                raise ValidationError("video_split_ratio_invalid", "At least one time-block split ratio must be greater than zero.")
        return {
            "sampling_mode": mode,
            "sampling_value": int(value) if mode == "frame_interval" else value,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "split_strategy": split_strategy,
            "time_block_seconds": time_block_seconds if split_strategy == "time_blocks" else None,
            "train_ratio": train_ratio if split_strategy == "time_blocks" else None,
            "val_ratio": val_ratio if split_strategy == "time_blocks" else None,
            "test_ratio": test_ratio if split_strategy == "time_blocks" else None,
        }

    def _validate_preflight(self, video_info: dict, config: dict) -> None:
        duration = float(video_info["duration_seconds"])
        if duration <= 0:
            raise ValidationError("video_duration_invalid", "The video duration could not be determined.")
        if duration > self.settings.max_video_duration_seconds:
            raise ValidationError("video_duration_too_long", "Video exceeds the configured duration limit.")
        if config["start_seconds"] >= duration:
            raise ValidationError("video_start_out_of_range", "Start time must be inside the video duration.")
        if config["end_seconds"] is not None and config["end_seconds"] > duration:
            config["end_seconds"] = duration
        estimated_images = estimate_images(video_info, config)
        if estimated_images > self.settings.max_video_generated_images:
            raise ValidationError("video_frame_limit_exceeded", "Sampling would generate more frames than the configured limit.")


def inspect_video(path: Path, source_file_name: str, size_bytes: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValidationError("video_unreadable", "The uploaded file is not a readable video.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValidationError("video_metadata_invalid", "Video metadata is incomplete or invalid.")
    return {
        "file_name": Path(source_file_name).name or "video",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "size_bytes": size_bytes,
    }


def estimate_images(video_info: dict, config: dict) -> int:
    fps = float(video_info["fps"])
    start = float(config["start_seconds"])
    end = float(config["end_seconds"] if config["end_seconds"] is not None else video_info["duration_seconds"])
    duration = max(end - start, 0.0)
    mode = config["sampling_mode"]
    value = float(config["sampling_value"])
    if mode == "fps":
        return max(1, math.ceil(duration * value))
    if mode == "interval_seconds":
        return max(1, math.ceil(duration / value))
    start_frame = math.floor(start * fps)
    end_frame = math.ceil(end * fps)
    return max(1, math.ceil(max(end_frame - start_frame, 0) / value))


def estimate_output_bytes(path: Path, video_info: dict, config: dict, estimated_images: int) -> tuple[int, int]:
    """Estimate JPEG output from evenly-spaced decoded samples before queueing work."""

    fps = float(video_info["fps"])
    start = float(config["start_seconds"])
    end = float(config["end_seconds"] if config["end_seconds"] is not None else video_info["duration_seconds"])
    sample_count = min(12, estimated_images)
    if sample_count <= 0:
        return 0, 0
    capture = cv2.VideoCapture(str(path))
    sizes: list[int] = []
    try:
        if not capture.isOpened():
            raise ValidationError("video_unreadable", "The uploaded file is not a readable video.")
        for index in range(sample_count):
            position = start + (end - start) * (index + 0.5) / sample_count
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, math.floor(position * fps)))
            ok, frame = capture.read()
            if not ok:
                continue
            encoded, jpeg = cv2.imencode(".jpg", frame)
            if encoded:
                sizes.append(len(jpeg))
    finally:
        capture.release()
    if not sizes:
        raise ValidationError("video_estimate_unavailable", "Could not decode representative frames for output-size estimation.")
    return math.ceil(sum(sizes) / len(sizes) * estimated_images), len(sizes)


def estimate_split_counts(video_info: dict, config: dict, estimated_images: int) -> dict[str, int]:
    if config["split_strategy"] == "single":
        return {"train": estimated_images if config.get("split") == "train" else 0, "val": estimated_images if config.get("split") == "val" else 0, "test": estimated_images if config.get("split") == "test" else 0}
    start = float(config["start_seconds"])
    end = float(config["end_seconds"] if config["end_seconds"] is not None else video_info["duration_seconds"])
    block_size = float(config["time_block_seconds"])
    block_count = max(1, math.ceil((end - start) / block_size))
    allocations = time_block_allocations(block_count, config)
    return {name: round(estimated_images * blocks / block_count) for name, blocks in allocations.items()}


def time_block_allocations(block_count: int, config: dict) -> dict[str, int]:
    names = ("train", "val", "test")
    raw = {name: block_count * float(config[f"{name}_ratio"]) for name in names}
    result = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(names, key=lambda item: (raw[item] - result[item], item), reverse=True)[: block_count - sum(result.values())]:
        result[name] += 1
    return result


def split_for_timestamp(timestamp: float, config: dict) -> str:
    if config["split_strategy"] == "single":
        return config["split"]
    start = float(config["start_seconds"])
    end = float(config["end_seconds"])
    block_count = max(1, math.ceil((end - start) / float(config["time_block_seconds"])))
    block_index = min(block_count - 1, max(0, math.floor((timestamp - start) / float(config["time_block_seconds"]))))
    allocations = time_block_allocations(block_count, config)
    if block_index < allocations["train"]:
        return "train"
    if block_index < allocations["train"] + allocations["val"]:
        return "val"
    return "test"
