from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.dataset.quality.schemas import ClassDistributionItem, DatasetQualityReport, QualityIssue, QualitySummary


class DatasetQualityService:
    """Starter adaptation of upstream DatasetQualityService; local schema has no auto-confidence fields."""

    def report(self, session: Session, dataset_id: str) -> DatasetQualityReport:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            from app.core.errors import NotFoundError
            raise NotFoundError("dataset_not_found", "Dataset was not found.")
        images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.file_name)))
        classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
        annotations = list(session.scalars(select(Annotation).where(Annotation.dataset_id == dataset_id)))
        return DatasetQualityReport(
            dataset_id=dataset_id,
            task_type=dataset.task_type,
            summary=self._summary(images, annotations),
            class_distribution=self._class_distribution(classes, annotations),
            issues=self._collect_issues(images, classes, annotations),
        )

    def _summary(self, images: list[ImageItem], annotations: list[Annotation]) -> QualitySummary:
        annotated_ids = {annotation.image_id for annotation in annotations}
        bbox = [annotation for annotation in annotations if annotation.type == "bbox"]
        small = sum(1 for annotation in bbox if self._bbox_area(annotation) < 32 * 32)
        total = len(images)
        annotated = len(annotated_ids.intersection({image.id for image in images}))
        return QualitySummary(image_count=total, annotated_image_count=annotated, unannotated_image_count=max(total - annotated, 0), coverage=round(annotated / total, 6) if total else 0, annotation_count=len(annotations), bbox_count=len(bbox), polygon_count=sum(item.type == "polygon" for item in annotations), obb_count=sum(item.type == "obb" for item in annotations), classify_count=sum(item.type == "classify" for item in annotations), small_object_count=small, small_object_ratio=round(small / len(bbox), 6) if bbox else 0)

    def _class_distribution(self, classes: list[ClassLabel], annotations: list[Annotation]) -> list[ClassDistributionItem]:
        counts = Counter(annotation.class_id for annotation in annotations)
        total = sum(counts.values())
        return [ClassDistributionItem(class_id=item.id, class_index=item.class_index, name=item.name, count=counts[item.id], ratio=round(counts[item.id] / total, 6) if total else 0) for item in classes]

    def _collect_issues(self, images: list[ImageItem], classes: list[ClassLabel], annotations: list[Annotation]) -> list[QualityIssue]:
        by_image: dict[str, list[Annotation]] = {}
        for item in annotations:
            by_image.setdefault(item.image_id, []).append(item)
        issues: list[QualityIssue] = []
        for item in annotations:
            if item.type == "bbox" and self._bbox_area(item) < 32 * 32:
                issues.append(QualityIssue(level="warning", type="small_object", image_id=item.image_id, annotation_ids=[item.id], class_id=item.class_id, value=self._bbox_area(item), message="Small bbox may be difficult for training."))
        for image_id, image_annotations in by_image.items():
            boxes = [item for item in image_annotations if item.type == "bbox"]
            for index, left in enumerate(boxes):
                for right in boxes[index + 1:]:
                    score = self._bbox_iou(left, right)
                    if left.class_id == right.class_id and score >= 0.85:
                        issues.append(QualityIssue(level="warning", type="similar_bbox", image_id=image_id, annotation_ids=[left.id, right.id], class_id=left.class_id, iou=round(score, 6), message="Two bbox annotations are highly overlapping."))
        counts = Counter(item.class_id for item in annotations)
        nonzero = [counts[item.id] for item in classes if counts[item.id]]
        if len(nonzero) >= 2:
            maximum = max(nonzero)
            for item in classes:
                if 0 < counts[item.id] / maximum < 0.1:
                    issues.append(QualityIssue(level="warning", type="class_imbalance", class_id=item.id, value=round(counts[item.id] / maximum, 6), message=f"Class {item.name} has much fewer annotations than the dominant class."))
        return issues

    @staticmethod
    def _bbox_area(annotation: Annotation) -> float:
        return float((annotation.width or 0) * (annotation.height or 0))

    @classmethod
    def _bbox_iou(cls, left: Annotation, right: Annotation) -> float:
        lx, ly, lw, lh = float(left.x or 0), float(left.y or 0), float(left.width or 0), float(left.height or 0)
        rx, ry, rw, rh = float(right.x or 0), float(right.y or 0), float(right.width or 0), float(right.height or 0)
        intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
        union = lw * lh + rw * rh - intersection
        return intersection / union if union > 0 else 0.0
