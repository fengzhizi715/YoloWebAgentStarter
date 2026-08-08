from app.core.errors import ValidationError
from app.core.schemas import BBox
from app.dataset.annotation.geometry import validate_bbox, validate_polygon


def test_bbox_is_kept_inside_image():
    validate_bbox(BBox(x=0, y=0, width=10, height=10), 20, 20)


def test_polygon_requires_area():
    try:
        validate_polygon([(0, 0), (1, 0), (0, 1)], 20, 20)
    except ValidationError as error:
        assert error.error_code == "polygon_too_small"
    else:
        raise AssertionError("expected polygon_too_small")
