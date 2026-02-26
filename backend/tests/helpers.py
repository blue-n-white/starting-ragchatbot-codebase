"""Shared test helpers for building mock Anthropic responses."""

from unittest.mock import MagicMock
from dataclasses import dataclass


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = "call_123"
    name: str = "search_course_content"
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {"query": "test"}


def make_text_response(text: str, stop_reason: str = "end_turn"):
    """Build a fake Anthropic messages.create() return value with text only."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = [TextBlock(text=text)]
    return resp


def make_tool_use_response(tool_name: str = "search_course_content",
                           tool_input: dict = None,
                           tool_id: str = "call_123"):
    """Build a fake Anthropic response that requests a tool call."""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [
        ToolUseBlock(id=tool_id, name=tool_name, input=tool_input or {"query": "test"}),
    ]
    return resp
