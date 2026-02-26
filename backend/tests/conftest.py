"""Shared pytest fixtures for RAG chatbot tests."""

import sys
import os
from unittest.mock import MagicMock

import pytest

# Add tests directory and backend directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from helpers import make_text_response


# ---------------------------------------------------------------------------
# VectorStore mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    """A MagicMock standing in for VectorStore with sensible defaults."""
    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=["Chunk A text", "Chunk B text"],
        metadata=[
            {"course_title": "Intro to AI", "lesson_number": 1},
            {"course_title": "Intro to AI", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )
    store.get_lesson_link.return_value = "https://example.com/lesson"
    return store


# ---------------------------------------------------------------------------
# CourseSearchTool + ToolManager fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_tool(mock_vector_store):
    """A CourseSearchTool wired to the mock vector store."""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(mock_vector_store):
    """A ToolManager with a CourseSearchTool registered."""
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_vector_store))
    return tm


# ---------------------------------------------------------------------------
# Anthropic client mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_client():
    """A MagicMock for anthropic.Anthropic with a messages.create stub."""
    client = MagicMock()
    client.messages.create.return_value = make_text_response("Hello from Claude")
    return client
