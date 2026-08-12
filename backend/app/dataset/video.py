from __future__ import annotations

import os
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.schemas import SplitName, VideoImportResponse
from app.core.storage import Storage
from app.dataset.images import add_uploaded_images
from app.dataset.service import get_dataset


def import_video_frames(session: Session, storage: Storage, dataset_id: str, content: bytes, suffix: str, split: SplitName, frame_interval: int) -> VideoImportResponse:
    """Synchronous Starter adaptation of the upstream OpenCV extraction loop, capped for local use."""
    get_dataset(session, dataset_id)
    if frame_interval < 1:
        raise ValidationError("video_config_invalid", "Frame interval must be at least 1.")
    try:
        import cv2
    except ImportError as exc:
        raise ValidationError("video_dependency_missing", "Video import requires OpenCV in the project .venv.") from exc
    descriptor, temporary = tempfile.mkstemp(suffix=suffix if suffix in {".mp4", ".mov", ".avi"} else ".mp4")
    os.close(descriptor)
    path = Path(temporary)
    path.write_bytes(content)
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValidationError("video_unreadable", "Could not open the uploaded video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[tuple[str, bytes]] = []
        index = 0
        while len(frames) < 1000:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_interval == 0:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise ValidationError("frame_write_failed", f"Failed to encode frame {index}.")
                frames.append((f"frame_{len(frames) + 1:06d}.jpg", bytes(buffer)))
            index += 1
        capture.release()
        if index < total and len(frames) >= 1000:
            raise ValidationError("video_frame_limit", "Video extraction exceeds the 1,000-frame Starter limit; use a larger frame interval.")
        add_uploaded_images(session, storage, dataset_id, frames, split)
        return VideoImportResponse(imported=len(frames), source_fps=fps, frame_count=total)
    finally:
        path.unlink(missing_ok=True)
