"""Tests for RAGSystem.query() integration."""

from unittest.mock import MagicMock, patch, call
from helpers import make_text_response, make_tool_use_response


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() with mocked components."""

    def _make_rag(self):
        """Build a RAGSystem with all heavy dependencies mocked out."""
        # Patch the imports used inside rag_system.py
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.SessionManager") as MockSM, \
             patch("rag_system.ToolManager") as MockTM, \
             patch("rag_system.CourseSearchTool") as MockCST, \
             patch("rag_system.CourseOutlineTool") as MockCOT:

            # Configure mock config
            config = MagicMock()
            config.DEMO_MODE = False
            config.ANTHROPIC_API_KEY = "fake-key"
            config.ANTHROPIC_MODEL = "claude-test"
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_db"
            config.EMBEDDING_MODEL = "test-model"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2

            from rag_system import RAGSystem
            rag = RAGSystem(config)

        return rag

    def test_query_passes_tools_to_generator(self):
        """query() passes tool definitions and tool_manager to AI generator."""
        rag = self._make_rag()
        rag.tool_manager.get_tool_definitions.return_value = [{"name": "search"}]
        rag.tool_manager.get_last_sources.return_value = []
        rag.ai_generator.generate_response.return_value = "answer"

        rag.query("What is AI?", session_id="s1")

        call_kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["tools"] == [{"name": "search"}]
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_query_returns_response_and_sources(self):
        """Return value is (response_string, sources_list)."""
        rag = self._make_rag()
        rag.ai_generator.generate_response.return_value = "The answer"
        rag.tool_manager.get_last_sources.return_value = [
            {"text": "Course A", "link": "http://a"}
        ]

        response, sources = rag.query("question")

        assert response == "The answer"
        assert sources == [{"text": "Course A", "link": "http://a"}]

    def test_query_collects_sources_from_tool_manager(self):
        """Sources come from tool_manager.get_last_sources()."""
        rag = self._make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        expected_sources = [{"text": "src", "link": None}]
        rag.tool_manager.get_last_sources.return_value = expected_sources

        _, sources = rag.query("q")

        rag.tool_manager.get_last_sources.assert_called_once()
        assert sources is expected_sources

    def test_query_resets_sources_after_retrieval(self):
        """tool_manager.reset_sources() is called after getting sources."""
        rag = self._make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("q")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_updates_session_history(self):
        """Session manager gets the exchange added when session_id is provided."""
        rag = self._make_rag()
        rag.ai_generator.generate_response.return_value = "the answer"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("my question", session_id="sess1")

        rag.session_manager.add_exchange.assert_called_once_with(
            "sess1", "my question", "the answer"
        )

    def test_query_no_session_skips_history_update(self):
        """When no session_id is provided, session manager is not called."""
        rag = self._make_rag()
        rag.ai_generator.generate_response.return_value = "resp"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("q", session_id=None)

        rag.session_manager.add_exchange.assert_not_called()

    def test_query_demo_mode_bypasses_ai(self):
        """Demo mode calls tool_manager directly instead of AI generator."""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator"), \
             patch("rag_system.SessionManager"), \
             patch("rag_system.ToolManager") as MockTM, \
             patch("rag_system.CourseSearchTool"), \
             patch("rag_system.CourseOutlineTool"):

            config = MagicMock()
            config.DEMO_MODE = True
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_db"
            config.EMBEDDING_MODEL = "test-model"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2

            from rag_system import RAGSystem
            rag = RAGSystem(config)

        # ai_generator should be None in demo mode
        assert rag.ai_generator is None

        rag.tool_manager.execute_tool.return_value = "demo search results"
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("test query")

        rag.tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )
        assert "Demo Mode" in response
        assert "demo search results" in response


class TestRAGSystemFullPipeline:
    """Minimal-mocking end-to-end test through the query pipeline."""

    def test_full_pipeline_tool_use_flow(self):
        """End-to-end: query -> AI -> tool_use -> execute -> results -> AI -> final answer."""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.SessionManager") as MockSM, \
             patch("rag_system.CourseSearchTool") as MockCST, \
             patch("rag_system.CourseOutlineTool") as MockCOT, \
             patch("rag_system.ToolManager") as MockTM:

            config = MagicMock()
            config.DEMO_MODE = False
            config.ANTHROPIC_API_KEY = "fake"
            config.ANTHROPIC_MODEL = "test"
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test_db"
            config.EMBEDDING_MODEL = "test"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2

            from rag_system import RAGSystem
            rag = RAGSystem(config)

        # Wire up the tool_manager mock
        rag.tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        rag.tool_manager.get_last_sources.return_value = [
            {"text": "Intro to AI - Lesson 1", "link": "https://example.com"}
        ]

        # AI returns a response
        rag.ai_generator.generate_response.return_value = "MCP stands for Model Context Protocol"

        response, sources = rag.query("What is MCP?", session_id="s1")

        # Verify the full flow
        assert response == "MCP stands for Model Context Protocol"
        assert len(sources) == 1
        assert sources[0]["text"] == "Intro to AI - Lesson 1"

        # AI generator was called with tools
        gen_call = rag.ai_generator.generate_response.call_args
        assert gen_call.kwargs["tools"] == [{"name": "search_course_content"}]
        assert gen_call.kwargs["tool_manager"] is rag.tool_manager

        # Sources were collected and reset
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()

        # Session was updated
        rag.session_manager.add_exchange.assert_called_once()
