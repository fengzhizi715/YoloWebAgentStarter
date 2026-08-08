from __future__ import annotations

import math

from app.core.errors import ValidationError
from app.core.schemas import BBox

MIN_SHAPE_SIZE = 3.0


def validate_bbox(bbox: BBox, image_width: int, image_height: int) -> None:
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("bbox_not_finite", "Bounding-box values must be finite.")
    if bbox.width < MIN_SHAPE_SIZE or bbox.height < MIN_SHAPE_SIZE:
        raise ValidationError("bbox_too_small", "Bounding boxes must be at least 3 × 3 pixels.")
    if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.width > image_width or bbox.y + bbox.height > image_height:
        raise ValidationError("bbox_out_of_bounds", "Bounding box must stay inside the image.")


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        / 2
    )


def validate_polygon(points: list[tuple[float, float]], image_width: int, image_height: int) -> None:
    if len(points) < 3:
        raise ValidationError("polygon_too_few_points", "Polygons require at least three points.")
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValidationError("polygon_not_finite", "Polygon coordinates must be finite.")
    if any(x < 0 or y < 0 or x > image_width or y > image_height for x, y in points):
        raise ValidationError("polygon_out_of_bounds", "Polygon points must stay inside the image.")
    if polygon_area(points) < MIN_SHAPE_SIZE * MIN_SHAPE_SIZE:
        raise ValidationError("polygon_too_small", "Polygon area is too small.")

