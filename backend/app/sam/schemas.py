from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.schemas import BBox


class SamPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1] = 1


class SamPredictRequest(BaseModel):
    image_id: str
    class_id: str
    prompt_type: Literal["box", "point"] = "box"
    box: BBox | None = None
    points: list[SamPoint] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def prompt_matches_type(self) -> "SamPredictRequest":
        if self.prompt_type == "box" and self.box is None:
            raise ValueError("box is required for a box prompt")
        if self.prompt_type == "point" and not self.points:
            raise ValueError("points are required for a point prompt")
        return self


class SamPredictResponse(BaseModel):
    image_id: str
    class_id: str
    mask_id: str
    polygon: list[tuple[float, float]]
    score: float
    backend_used: str
    device: str | None = None
