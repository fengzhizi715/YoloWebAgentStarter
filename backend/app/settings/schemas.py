from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SamFallbackMode = Literal["box", "disabled"]
LLMAuthScheme = Literal["bearer", "header", "raw_authorization"]


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


# Adapted from upstream YoloWebAgent settings/schemas.py (Community subset).
class LLMSettingsOut(BaseModel):
    enabled: bool = False
    provider: str = "openai-compatible"
    api_base: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_seconds: int = 60
    api_key_configured: bool = False
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMSettingsUpdate(BaseModel):
    enabled: bool = False
    provider: str = "openai-compatible"
    api_base: str = ""
    api_key: str | None = Field(default=None, description="Leave empty to keep the existing key.")
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=1, le=7200)
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMConnectionTestRequest(BaseModel):
    """Test connectivity without persisting settings (upstream pattern)."""

    api_base: str = ""
    model: str = "gpt-4o-mini"
    api_key: str | None = Field(default=None, description="Optional; empty uses saved key.")
    timeout_seconds: int = Field(default=60, ge=1, le=7200)
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMConnectionTestResult(BaseModel):
    ok: bool
    message: str
    remote_test_performed: bool = False
    # Stable machine code for UI localization (message remains Chinese for logs/compat).
    code: str = "ok"
    http_status: int | None = None


class LLMSettingsInternal(LLMSettingsOut):
    api_key: str = ""
