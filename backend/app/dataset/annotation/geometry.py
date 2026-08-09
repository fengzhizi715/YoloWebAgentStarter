from __future__ import annotations

import math

from app.core.errors import ValidationError
from app.core.schemas import BBox, OBB

MIN_SHAPE_SIZE = 3.0
BOUNDARY_EPSILON = 1e-6


def validate_bbox(bbox: BBox, image_width: int, image_height: int) -> None:
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("bbox_not_finite", "Bounding-box values must be finite.")
    if bbox.width < MIN_SHAPE_SIZE or bbox.height < MIN_SHAPE_SIZE:
        raise ValidationError("bbox_too_small", "Bounding boxes must be at least 3 × 3 pixels.")
    if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.width > image_width or bbox.y + bbox.height > image_height:
        raise ValidationError("bbox_out_of_bounds", "Bounding box must stay inside the image.")


def obb_corners(obb: OBB) -> list[tuple[float, float]]:
    """Return the four OBB corners in clockwise order.

    The representation intentionally matches YoloWebAgent's persisted OBB
    contract: center and dimensions in pixels, with an angle in degrees.
    """

    radians = math.radians(obb.angle)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    half_width = obb.width / 2
    half_height = obb.height / 2
    return [
        (
            obb.cx + offset_x * cos_angle - offset_y * sin_angle,
            obb.cy + offset_x * sin_angle + offset_y * cos_angle,
        )
        for offset_x, offset_y in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        )
    ]


def validate_obb(obb: OBB, image_width: int, image_height: int) -> None:
    values = (obb.cx, obb.cy, obb.width, obb.height, obb.angle)
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("obb_not_finite", "Oriented bounding-box values must be finite.")
    if obb.width < MIN_SHAPE_SIZE or obb.height < MIN_SHAPE_SIZE:
        raise ValidationError("obb_too_small", "Oriented bounding boxes must be at least 3 × 3 pixels.")
    # Trigonometric reconstruction of a right-angle OBB can produce values
    # such as -3e-15 at an otherwise exact image boundary.  Accept only that
    # numerical noise; meaningful out-of-bounds geometry remains rejected.
    if any(
        x < -BOUNDARY_EPSILON
        or y < -BOUNDARY_EPSILON
        or x > image_width + BOUNDARY_EPSILON
        or y > image_height + BOUNDARY_EPSILON
        for x, y in obb_corners(obb)
    ):
        raise ValidationError("obb_out_of_bounds", "Oriented bounding box must stay inside the image.")


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
