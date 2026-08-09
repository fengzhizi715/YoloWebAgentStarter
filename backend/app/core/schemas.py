from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.task_types import TaskType

SplitName = Literal["train", "val", "test"]
AnnotationType = Literal["bbox", "polygon", "obb", "classify"]
AnnotationSource = Literal["manual", "imported", "sam"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    task_type: TaskType

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip()


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def strip_optional_name(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class DatasetResponse(ORMModel):
    id: str
    name: str
    description: str | None
    task_type: TaskType
    image_count: int
    class_count: int
    created_at: datetime
    updated_at: datetime


class ClassLabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    color: str = Field(default="#22c55e", pattern=r"^#[0-9a-fA-F]{6}$")

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("class name cannot be blank")
        if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
            raise ValueError("class name cannot contain path separators")
        return cleaned


class ClassLabelResponse(ORMModel):
    id: str
    dataset_id: str
    class_index: int
    name: str
    color: str
    created_at: datetime
    updated_at: datetime


class ScanImagesRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)
    recursive: bool = True
    split: SplitName = "train"


class ScanImagesResponse(BaseModel):
    total_found: int
    imported: int
    skipped: int
    invalid: int


class ImageSplitUpdate(BaseModel):
    split: SplitName


class ImageItemResponse(ORMModel):
    id: str
    dataset_id: str
    file_name: str
    width: int
    height: int
    split: SplitName
    status: str
    file_url: str
    created_at: datetime
    updated_at: datetime


class ImagePage(BaseModel):
    items: list[ImageItemResponse]
    total: int


class UploadImagesResponse(BaseModel):
    imported: int
    items: list[ImageItemResponse]


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OBB(BaseModel):
    """An oriented bounding box stored in absolute image pixels."""

    cx: float
    cy: float
    width: float
    height: float
    angle: float


class AnnotationInput(BaseModel):
    type: AnnotationType
    class_id: str
    bbox: BBox | None = None
    polygon: list[tuple[float, float]] | None = None
    obb: OBB | None = None
    source: AnnotationSource = "manual"

    @model_validator(mode="after")
    def shape_matches_type(self) -> "AnnotationInput":
        if self.type == "bbox" and self.bbox is None:
            raise ValueError("bbox is required for bbox annotations")
        if self.type == "polygon" and self.polygon is None:
            raise ValueError("polygon is required for polygon annotations")
        if self.type == "obb" and self.obb is None:
            raise ValueError("obb is required for oriented bounding-box annotations")
        return self

    @field_validator("polygon")
    @classmethod
    def finite_polygon(cls, value: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
        if value is not None and not all(math.isfinite(x) and math.isfinite(y) for x, y in value):
            raise ValueError("polygon coordinates must be finite")
        return value


class ReplaceAnnotationsRequest(BaseModel):
    annotations: list[AnnotationInput] = Field(max_length=10000)


class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    dataset_id: str
    class_id: str
    class_index: int
    label: str
    color: str
    type: AnnotationType
    bbox: BBox | None = None
    polygon: list[tuple[float, float]] | None = None
    obb: OBB | None = None
    source: AnnotationSource
    created_at: datetime
    updated_at: datetime


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    code: str
    message: str
    image_id: str | None = None
    annotation_id: str | None = None


class ValidationReport(BaseModel):
    dataset_id: str
    valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssue]


class YoloImportResponse(BaseModel):
    dataset: DatasetResponse
    imported_images: int
    imported_annotations: int


class DatasetDetailResponse(DatasetResponse):
    classes: list[ClassLabelResponse]
    image_total: int


class YoloExportInfo(BaseModel):
    dataset_id: str
    file_name: str
