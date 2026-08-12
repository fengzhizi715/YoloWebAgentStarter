from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from app.core.errors import ValidationError

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class Storage:
    def __init__(self, data_dir: Path, import_root: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.import_root = import_root.resolve()
        self.datasets_dir = self.data_dir / "datasets"
        self.exports_dir = self.data_dir / "exports"
        self.training_dir = self.data_dir / "runs" / "training"
        self.evaluation_dir = self.data_dir / "runs" / "evaluation"
        self.models_dir = self.data_dir / "models"
        self.tmp_dir = self.data_dir / "tmp"

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def images_dir(self, dataset_id: str) -> Path:
        path = self.dataset_dir(dataset_id) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def image_path(self, dataset_id: str, storage_name: str) -> Path:
        candidate = (self.images_dir(dataset_id) / storage_name).resolve()
        if not _is_within(candidate, self.images_dir(dataset_id).resolve()):
            raise ValidationError("unsafe_path", "Image path escapes the managed dataset directory.")
        return candidate

    def resolve_import_directory(self, requested: str) -> Path:
        raw = Path(requested).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (self.import_root / raw).resolve()
        if not _is_within(candidate, self.import_root):
            raise ValidationError("import_path_outside_root", "Import path must be inside YWA_IMPORT_ROOT.")
        if not candidate.is_dir():
            raise ValidationError("import_directory_missing", "Import directory does not exist.")
        return candidate

    def resolve_import_file(self, candidate: Path) -> Path:
        """Resolve one scanned file without allowing symlinks to escape the import root."""

        resolved = candidate.resolve()
        if not _is_within(resolved, self.import_root):
            raise ValidationError("import_path_outside_root", "Import file must stay inside YWA_IMPORT_ROOT.")
        if not resolved.is_file():
            raise ValidationError("import_file_missing", "Import file does not exist.")
        return resolved

    def safe_storage_name(self, original_name: str, image_id: str) -> str:
        source = Path(original_name).name
        suffix = Path(source).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValidationError("unsupported_image_format", f"Unsupported image format: {suffix or 'unknown'}.")
        stem = _SAFE_NAME.sub("_", Path(source).stem).strip("._") or "image"
        return f"{image_id}_{stem}{suffix}"

    def _verify_image(self, destination: Path) -> tuple[int, int]:
        """Validate a newly written managed image and return its dimensions."""

        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            destination.unlink(missing_ok=True)
            raise ValidationError("invalid_image", "The uploaded file is not a readable image.") from exc
        if width <= 0 or height <= 0:
            destination.unlink(missing_ok=True)
            raise ValidationError("invalid_image_size", "Image dimensions must be greater than zero.")
        return width, height

    def write_image_stream(
        self,
        dataset_id: str,
        storage_name: str,
        source: BinaryIO,
        *,
        max_bytes: int,
        chunk_bytes: int = 1024 * 1024,
    ) -> tuple[int, int]:
        """Stream an image into managed storage without buffering the source in memory."""

        destination = self.image_path(dataset_id, storage_name)
        try:
            total = 0
            with destination.open("wb") as target:
                while chunk := source.read(chunk_bytes):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValidationError("image_too_large", "Image exceeds the configured archive member limit.")
                    target.write(chunk)
        except (OSError, ValidationError):
            destination.unlink(missing_ok=True)
            raise
        return self._verify_image(destination)

    def write_image(self, dataset_id: str, storage_name: str, content: bytes) -> tuple[int, int]:
        return self.write_image_stream(dataset_id, storage_name, io.BytesIO(content), max_bytes=len(content))

    def read_image(self, dataset_id: str, storage_name: str) -> bytes:
        path = self.image_path(dataset_id, storage_name)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ValidationError("image_read_failed", "Could not read the managed image file.") from exc

    def copy_image(self, source: Path, dataset_id: str, storage_name: str) -> tuple[int, int]:
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise ValidationError("image_read_failed", f"Could not read {source.name}.") from exc
        return self.write_image(dataset_id, storage_name, content)

    def remove_dataset(self, dataset_id: str) -> None:
        directory = self.dataset_dir(dataset_id).resolve()
        if _is_within(directory, self.datasets_dir.resolve()) and directory.exists():
            shutil.rmtree(directory)

    def export_path(self, file_name: str) -> Path:
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        candidate = (self.exports_dir / Path(file_name).name).resolve()
        if not _is_within(candidate, self.exports_dir.resolve()):
            raise ValidationError("unsafe_path", "Export path escapes the managed export directory.")
        return candidate

    def training_task_dir(self, task_id: str) -> Path:
        candidate = (self.training_dir / task_id).resolve()
        if not _is_within(candidate, self.training_dir.resolve()):
            raise ValidationError("unsafe_path", "Training task path escapes the managed training directory.")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def managed_training_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        if not _is_within(candidate, self.training_dir.resolve()):
            raise ValidationError("unsafe_path", "Training path must stay inside the managed training directory.")
        return candidate

    def evaluation_task_dir(self, evaluation_id: str) -> Path:
        candidate = (self.evaluation_dir / evaluation_id).resolve()
        if not _is_within(candidate, self.evaluation_dir.resolve()):
            raise ValidationError("unsafe_path", "Evaluation task path escapes the managed evaluation directory.")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def managed_evaluation_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        if not _is_within(candidate, self.evaluation_dir.resolve()):
            raise ValidationError("unsafe_path", "Evaluation path must stay inside the managed evaluation directory.")
        return candidate

    def remove_evaluation_task(self, evaluation_id: str) -> None:
        directory = (self.evaluation_dir / evaluation_id).resolve()
        if _is_within(directory, self.evaluation_dir.resolve()) and directory.exists():
            shutil.rmtree(directory)

    def model_version_dir(self, model_id: str) -> Path:
        candidate = (self.models_dir / model_id).resolve()
        if not _is_within(candidate, self.models_dir.resolve()):
            raise ValidationError("unsafe_path", "Model path escapes the managed model directory.")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def model_test_path(self, model_id: str, record_id: str, file_name: str) -> Path:
        directory = self.model_version_dir(model_id) / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        candidate = (directory / f"{record_id}_{Path(file_name).name}").resolve()
        if not _is_within(candidate, directory.resolve()):
            raise ValidationError("unsafe_path", "Model test image path escapes managed storage.")
        return candidate

    def write_model_test_image(self, model_id: str, record_id: str, file_name: str, content: bytes) -> Path:
        path = self.model_test_path(model_id, record_id, file_name)
        try:
            path.write_bytes(content)
        except OSError as exc:
            raise ValidationError("model_test_write_failed", "Could not store the managed model test image.") from exc
        return path

    def managed_model_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        if not _is_within(candidate, self.models_dir.resolve()):
            raise ValidationError("unsafe_path", "Model path must stay inside the managed model directory.")
        return candidate

    def copy_model_artifact(self, source: Path, model_id: str, file_name: str) -> Path:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise ValidationError("model_file_missing", "The model artifact does not exist on disk.")
        destination = (self.model_version_dir(model_id) / Path(file_name).name).resolve()
        if not _is_within(destination, self.model_version_dir(model_id).resolve()):
            raise ValidationError("unsafe_path", "Model artifact path escapes the managed model directory.")
        shutil.copy2(source, destination)
        return destination

    def remove_model_version(self, model_id: str) -> None:
        directory = self.model_version_dir(model_id)
        if directory.exists():
            shutil.rmtree(directory)

    def remove_training_task(self, task_id: str) -> None:
        directory = self.training_task_dir(task_id)
        if directory.exists():
            shutil.rmtree(directory)
