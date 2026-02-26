"""Tests for CourseSearchTool.execute() and ToolManager."""

from unittest.mock import MagicMock
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


# ── CourseSearchTool.execute() ────────────────────────────────────────────


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute()."""

    def test_execute_returns_formatted_results(self, search_tool, mock_vector_store):
        """Successful search returns formatted text with course/lesson headers."""
        result = search_tool.execute(query="neural networks")

        assert "[Intro to AI - Lesson 1]" in result
        assert "Chunk A text" in result
        assert "[Intro to AI - Lesson 2]" in result
        assert "Chunk B text" in result

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Empty results return a 'No relevant content found' message."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Empty results include course/lesson filter information."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(
            query="nonexistent", course_name="MCP", lesson_number=3
        )

        assert "No relevant content found" in result
        assert "MCP" in result
        assert "lesson 3" in result

    def test_execute_with_error(self, search_tool, mock_vector_store):
        """Error from vector store is returned as-is."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="Search error: timeout"
        )

        result = search_tool.execute(query="anything")

        assert result == "Search error: timeout"

    def test_execute_passes_filters_to_store(self, search_tool, mock_vector_store):
        """course_name and lesson_number are forwarded to store.search()."""
        search_tool.execute(query="test", course_name="AI Basics", lesson_number=5)

        mock_vector_store.search.assert_called_once_with(
            query="test", course_name="AI Basics", lesson_number=5
        )

    def test_execute_tracks_sources(self, search_tool, mock_vector_store):
        """last_sources is populated with text and link for each result."""
        search_tool.execute(query="test")

        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Intro to AI - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "https://example.com/lesson"
        assert search_tool.last_sources[1]["text"] == "Intro to AI - Lesson 2"


# ── ToolManager ───────────────────────────────────────────────────────────


class TestToolManager:
    """Tests for ToolManager registration, dispatch, and sources."""

    def test_tool_manager_registration(self, mock_vector_store):
        """ToolManager correctly registers and retrieves tool definitions."""
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        definitions = tm.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_manager_execute_dispatches(self, tool_manager):
        """ToolManager.execute_tool dispatches to the correct tool."""
        result = tool_manager.execute_tool(
            "search_course_content", query="test query"
        )

        # Should return formatted results (not an error string)
        assert "Chunk A text" in result

    def test_tool_manager_execute_unknown_tool(self, tool_manager):
        """Requesting an unknown tool returns an error string."""
        result = tool_manager.execute_tool("nonexistent_tool", query="x")

        assert "not found" in result

    def test_tool_manager_get_last_sources(self, tool_manager):
        """Sources are retrieved from the right tool after search."""
        tool_manager.execute_tool("search_course_content", query="test")

        sources = tool_manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]["text"] == "Intro to AI - Lesson 1"

    def test_tool_manager_reset_sources(self, tool_manager):
        """reset_sources() clears all tracked sources."""
        tool_manager.execute_tool("search_course_content", query="test")
        assert tool_manager.get_last_sources()  # non-empty

        tool_manager.reset_sources()

        assert tool_manager.get_last_sources() == []
