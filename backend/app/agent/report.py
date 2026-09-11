from __future__ import annotations

from typing import Any

from app.agent.reply_renderer import AgentReplyRenderer


_renderer = AgentReplyRenderer()


def format_tool_report(tool_results: list[dict[str, Any]]) -> str:
    """Turn bounded tool payloads into a short natural-language report for the mock provider."""
    return _renderer.format_tool_report(tool_results)
