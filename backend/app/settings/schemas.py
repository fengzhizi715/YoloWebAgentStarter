from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SamFallbackMode = Literal["box", "disabled"]


class SamSettingsResponse(BaseModel):
    enabled: bool
    model: str
    device: str
    img_size: int
    fallback_mode: SamFallbackMode
    model_configured: bool


class SamSettingsUpdate(BaseModel):
    enabled: bool = True
    model: str = Field(default="", max_length=1024)
    device: str = Field(default="auto", min_length=1, max_length=64)
    img_size: int = Field(default=1024, ge=64, le=4096)
    fallback_mode: SamFallbackMode = "box"
