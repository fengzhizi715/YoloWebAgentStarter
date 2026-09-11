from __future__ import annotations

import math
import shutil

import cv2
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.models import Dataset, ImageItem, VideoImportTask
from app.core.storage import Storage
from app.core.time import utc_now
from app.dataset.service import refresh_dataset_counts
from app.training.observability.log_store import TrainingLogStore
from app.video_import.service import split_for_timestamp


class VideoImportRunner:
    def __init__(self, session_factory: sessionmaker[Session], queue, storage: Storage, settings: Settings) -> None:
        self.session_factory = session_factory
        self.queue = queue
        self.storage = storage
        self.settings = settings

    def run(self, task_id: str) -> None:
        log_store: TrainingLogStore | None = None
        try:
            spec = self._claim(task_id)
            if spec is None:
                return
            log_store = TrainingLogStore(spec["logs_path"])
            log_store.append(f"Video import started: {spec['source_file_name']}.")
            self._extract(task_id, spec, log_store)
        except ValidationError as exc:
            if log_store:
                log_store.append(f"Video import failed: {exc.message}")
            self._fail(task_id, exc.message)
        except Exception:
            if log_store:
                log_store.append("Video import failed unexpectedly. Check runtime logs.")
            self._fail(task_id, "Video import failed unexpectedly. Check runtime logs.")
        finally:
            self.queue.on_finished(task_id)

    def _claim(self, task_id: str) -> dict | None:
        with self.session_factory() as session:
            task = session.get(VideoImportTask, task_id)
            if task is None or task.status != "pending":
                return None
            source_path = self.storage.video_import_source_path(task.id, task.source_storage_name)
            if not source_path.is_file():
                task.status = "failed"
                task.error_message = "The staged video source is missing; upload it again."
                session.commit()
                return None
            config = dict(task.config_json)
            config["end_seconds"] = config["end_seconds"] if config["end_seconds"] is not None else task.video_info_json["duration_seconds"]
            dataset = session.get(Dataset, task.dataset_id) if task.dataset_id else None
            if dataset is None:
                if task.checkpoint_next_frame_index or task.generated_images:
                    task.status = "failed"
                    task.error_message = "The partially imported dataset is missing and cannot be resumed. Retry to start over."
                    session.commit()
                    return None
                from app.core.ids import new_id

                dataset = Dataset(id=new_id("ds"), name=task.name, description=None, task_type=task.task_type)
                session.add(dataset)
                task.dataset_id = dataset.id
            estimated_remaining = max(0, int(task.video_info_json.get("estimated_output_bytes", 0)) - task.output_bytes)
            self._ensure_free_space(estimated_remaining)
            task.status = "running"
            task.error_message = None
            session.commit()
            return {
                "dataset_id": dataset.id,
                "source_path": source_path,
                "source_file_name": task.source_file_name,
                "source_checksum": task.source_checksum,
                "source_group_id": f"video:{task.id}",
                "config": config,
                "video_info": dict(task.video_info_json),
                "total_images": task.total_images,
                "generated_images": task.generated_images,
                "output_bytes": task.output_bytes,
                "checkpoint_next_frame_index": task.checkpoint_next_frame_index,
                "checkpoint_next_sample_at": task.checkpoint_next_sample_at,
                "logs_path": task.logs_path or str(self.storage.video_import_task_dir(task.id) / "video_import.log"),
            }

    def _extract(self, task_id: str, spec: dict, log_store: TrainingLogStore) -> None:
        video_info = spec["video_info"]
        config = spec["config"]
        fps = float(video_info["fps"])
        start = float(config["start_seconds"])
        end = float(config["end_seconds"] if config["end_seconds"] is not None else video_info["duration_seconds"])
        start_frame = max(0, math.floor(start * fps))
        capture = cv2.VideoCapture(str(spec["source_path"]))
        output_bytes = int(spec["output_bytes"])
        generated = int(spec["generated_images"])
        try:
            if not capture.isOpened():
                raise ValidationError("video_unreadable", "The staged video is no longer readable.")
            frame_index = max(start_frame, int(spec["checkpoint_next_frame_index"]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            next_sample_at = float(spec["checkpoint_next_sample_at"] if spec["checkpoint_next_sample_at"] is not None else start)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                if timestamp >= end:
                    break
                if self._should_keep(config, timestamp, frame_index, start_frame, next_sample_at):
                    if config["sampling_mode"] in {"fps", "interval_seconds"}:
                        gap = 1 / float(config["sampling_value"]) if config["sampling_mode"] == "fps" else float(config["sampling_value"])
                        while next_sample_at <= timestamp + 1e-9:
                            next_sample_at += gap
                    encoded, jpeg = cv2.imencode(".jpg", frame)
                    if not encoded:
                        raise ValidationError("video_frame_encode_failed", "A selected video frame could not be encoded as JPEG.")
                    content = jpeg.tobytes()
                    output_bytes += len(content)
                    if output_bytes > self.settings.max_video_output_bytes:
                        raise ValidationError("video_output_limit_exceeded", "Generated frame data exceeds the configured output limit.")
                    image_id = f"img_{task_id[4:]}_{frame_index:09d}"
                    storage_name = self.storage.safe_storage_name(f"frame_{frame_index:09d}.jpg", image_id)
                    width, height = self.storage.write_image(spec["dataset_id"], storage_name, content)
                    with self.session_factory() as session:
                        task = session.get(VideoImportTask, task_id)
                        if task is None or task.status != "running":
                            raise RuntimeError("Video import task was stopped or removed.")
                        session.add(
                            ImageItem(
                                id=image_id,
                                dataset_id=spec["dataset_id"],
                                file_name=f"frame_{frame_index:09d}.jpg",
                                storage_name=storage_name,
                                width=width,
                                height=height,
                                split=split_for_timestamp(timestamp, config),
                                status="unannotated",
                                source_type="video_frame",
                                source_file=spec["source_file_name"],
                                source_group_id=spec["source_group_id"],
                                source_video_task_id=task_id,
                                source_checksum=spec["source_checksum"],
                                frame_index=frame_index,
                                timestamp=timestamp,
                            )
                        )
                        generated += 1
                        task.generated_images = generated
                        task.progress_percent = min(99.0, round(generated / max(spec["total_images"], 1) * 100, 2))
                        task.checkpoint_next_frame_index = frame_index + 1
                        task.checkpoint_next_sample_at = next_sample_at
                        task.output_bytes = output_bytes
                        session.commit()
                    if generated >= self.settings.max_video_generated_images:
                        raise ValidationError("video_frame_limit_exceeded", "Generated frame count reached the configured limit.")
                    if generated % 25 == 0:
                        self._ensure_free_space()
                        log_store.append(f"Generated {generated} frame(s).")
                frame_index += 1
            if generated == 0:
                raise ValidationError("video_no_frames_selected", "Sampling settings selected no frames from this video.")
            with self.session_factory() as session:
                task = session.get(VideoImportTask, task_id)
                if task is None:
                    return
                refresh_dataset_counts(session, spec["dataset_id"])
                task.generated_images = generated
                task.progress_percent = 100.0
                task.checkpoint_next_frame_index = frame_index
                task.checkpoint_next_sample_at = next_sample_at
                task.output_bytes = output_bytes
                task.status = "completed"
                task.error_message = None
                session.commit()
            log_store.append(f"Video import completed with {generated} frame(s). Review neighboring frames before training.")
        finally:
            capture.release()

    @staticmethod
    def _should_keep(config: dict, timestamp: float, frame_index: int, start_frame: int, next_sample_at: float) -> bool:
        mode = config["sampling_mode"]
        if mode == "frame_interval":
            return (frame_index - start_frame) % int(config["sampling_value"]) == 0
        return timestamp + 1e-9 >= next_sample_at

    def _ensure_free_space(self, expected_output_bytes: int = 0) -> None:
        free_bytes = shutil.disk_usage(self.storage.data_dir).free
        if free_bytes < self.settings.min_video_free_bytes + expected_output_bytes:
            raise ValidationError("video_insufficient_storage", "Not enough free disk space for video frame extraction.")

    def _fail(self, task_id: str, message: str) -> None:
        dataset_id: str | None = None
        with self.session_factory() as session:
            task = session.get(VideoImportTask, task_id)
            if task is None:
                return
            dataset_id = task.dataset_id
            task.dataset_id = None
            task.status = "failed"
            task.error_message = message
            if dataset_id:
                dataset = session.get(Dataset, dataset_id)
                if dataset is not None:
                    session.delete(dataset)
            session.commit()
        if dataset_id:
            self.storage.remove_dataset(dataset_id)
