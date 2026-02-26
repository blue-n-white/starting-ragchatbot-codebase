"""Tests for AIGenerator tool-calling flow."""

from unittest.mock import MagicMock, patch
from helpers import make_text_response, make_tool_use_response

from ai_generator import AIGenerator


def _make_generator(mock_client):
    """Build an AIGenerator with the Anthropic client already replaced."""
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_client):
        gen = AIGenerator(api_key="fake-key", model="claude-test")
    return gen


class TestAIGeneratorDirectResponse:
    """When Claude returns text without tool use."""

    def test_direct_response_no_tools(self, mock_anthropic_client):
        """Text-only response (no tool_use) is returned directly."""
        gen = _make_generator(mock_anthropic_client)

        result = gen.generate_response(query="What is Python?")

        assert result == "Hello from Claude"

    def test_tool_definitions_passed_to_api(self, mock_anthropic_client):
        """Tool definitions are included in the API call when provided."""
        gen = _make_generator(mock_anthropic_client)
        tools = [{"name": "search_course_content", "description": "...",
                   "input_schema": {"type": "object", "properties": {}}}]

        gen.generate_response(query="test", tools=tools)

        api_call = mock_anthropic_client.messages.create.call_args
        assert api_call.kwargs["tools"] == tools
        assert api_call.kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_in_system(self, mock_anthropic_client):
        """Conversation history is appended to the system prompt."""
        gen = _make_generator(mock_anthropic_client)
        history = "User: Hi\nAssistant: Hello!"

        gen.generate_response(query="test", conversation_history=history)

        api_call = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation:" in api_call.kwargs["system"]
        assert history in api_call.kwargs["system"]


class TestAIGeneratorToolUse:
    """When Claude requests a tool call."""

    def test_tool_use_triggers_execution(self, mock_anthropic_client):
        """stop_reason='tool_use' causes tool_manager.execute_tool to be called."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_name="search_course_content",
                                   tool_input={"query": "MCP"}),
            make_text_response("Here are the results"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "some search results"

        gen.generate_response(
            query="Tell me about MCP",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_mgr,
        )

        tool_mgr.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP"
        )

    def test_tool_result_sent_back_to_api(self, mock_anthropic_client):
        """Tool results are appended as a user message with tool_result format."""
        tool_use_resp = make_tool_use_response(tool_id="call_456")
        mock_anthropic_client.messages.create.side_effect = [
            tool_use_resp,
            make_text_response("Final answer"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "tool output text"

        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}],
            tool_manager=tool_mgr,
        )

        # Second call should have the tool_result in messages
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Last message should be from "user" containing tool results
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_use_id"] == "call_456"
        assert user_msg["content"][0]["content"] == "tool output text"

    def test_final_response_returned_after_tool(self, mock_anthropic_client):
        """After tool execution, the final Claude response text is returned."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response("The answer is 42"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "results"

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr
        )

        assert result == "The answer is 42"


class TestMultiToolRounds:
    """Tests for sequential multi-tool-call support."""

    def test_two_sequential_tool_calls(self, mock_anthropic_client):
        """Two tool rounds produce 3 API calls and 2 tool executions."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(
                tool_name="get_course_outline", tool_id="call_1",
                tool_input={"course_name": "MCP"}),
            make_tool_use_response(
                tool_name="search_course_content", tool_id="call_2",
                tool_input={"query": "lesson 4 topic"}),
            make_text_response("Final synthesized answer"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["outline results", "search results"]

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        assert mock_anthropic_client.messages.create.call_count == 3
        assert tool_mgr.execute_tool.call_count == 2
        assert result == "Final synthesized answer"

    def test_messages_accumulate_across_rounds(self, mock_anthropic_client):
        """The third API call contains the full message history from both rounds."""
        resp1 = make_tool_use_response(
            tool_name="get_course_outline", tool_id="call_1",
            tool_input={"course_name": "AI"})
        resp2 = make_tool_use_response(
            tool_name="search_course_content", tool_id="call_2",
            tool_input={"query": "deep learning"})
        mock_anthropic_client.messages.create.side_effect = [
            resp1, resp2, make_text_response("done"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["outline", "content"]

        gen.generate_response(
            query="user question", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        third_call = mock_anthropic_client.messages.create.call_args_list[2]
        messages = third_call.kwargs["messages"]

        # 5 messages: user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0] == {"role": "user", "content": "user question"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is resp1.content
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["tool_use_id"] == "call_1"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] is resp2.content
        assert messages[4]["role"] == "user"
        assert messages[4]["content"][0]["tool_use_id"] == "call_2"

    def test_tools_included_in_all_api_calls(self, mock_anthropic_client):
        """Tool definitions are present in every API call during the loop."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="call_1"),
            make_text_response("answer"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "r"
        tools = [{"name": "search_course_content"}]

        gen.generate_response(query="q", tools=tools, tool_manager=tool_mgr)

        for i, api_call in enumerate(mock_anthropic_client.messages.create.call_args_list):
            assert "tools" in api_call.kwargs, f"API call {i} missing tools"

    def test_loop_stops_at_max_rounds(self, mock_anthropic_client):
        """After MAX_TOOL_ROUNDS tool executions, the loop exits even if Claude wants more."""
        # 3 responses: initial + 2 follow-ups. The 3rd is a tool_use that won't be executed.
        from helpers import TextBlock, ToolUseBlock
        third_resp = MagicMock()
        third_resp.stop_reason = "tool_use"
        third_resp.content = [
            TextBlock(text="I want to search more"),
            ToolUseBlock(id="call_3", name="search", input={"query": "x"}),
        ]
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="call_1"),
            make_tool_use_response(tool_id="call_2"),
            third_resp,
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "r"

        result = gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        # initial + MAX_TOOL_ROUNDS follow-up calls = 3 API calls
        assert mock_anthropic_client.messages.create.call_count == 3
        # Only MAX_TOOL_ROUNDS tool executions (third tool_use is not executed)
        assert tool_mgr.execute_tool.call_count == 2
        # Returns text from the third response (loop exited, no more tool execution)
        assert result == "I want to search more"

    def test_single_tool_call_still_works(self, mock_anthropic_client):
        """One tool call followed by text produces 2 API calls and 1 tool execution."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="call_1"),
            make_text_response("the answer"),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "results"

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        assert mock_anthropic_client.messages.create.call_count == 2
        assert tool_mgr.execute_tool.call_count == 1
        assert result == "the answer"

    def test_no_tool_manager_skips_tool_execution(self, mock_anthropic_client):
        """When tool_manager is None, tool_use response text is returned directly."""
        from helpers import TextBlock, ToolUseBlock
        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [
            TextBlock(text="I want to search"),
            ToolUseBlock(id="call_1", name="search_course_content",
                         input={"query": "test"}),
        ]
        mock_anthropic_client.messages.create.return_value = resp
        gen = _make_generator(mock_anthropic_client)

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=None,
        )

        assert mock_anthropic_client.messages.create.call_count == 1
        assert result == "I want to search"

    def test_tool_error_string_sent_back_to_claude(self, mock_anthropic_client):
        """Tool error strings are passed to Claude as tool_result content."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="call_1"),
            make_text_response("Sorry, I couldn't find that."),
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "Tool 'bad_tool' not found"

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        tool_result_msg = second_call.kwargs["messages"][-1]
        assert tool_result_msg["content"][0]["content"] == "Tool 'bad_tool' not found"
        assert result == "Sorry, I couldn't find that."

    def test_fallback_when_no_text_block(self, mock_anthropic_client):
        """Returns fallback message when the final response has no text block."""
        from helpers import ToolUseBlock
        # All 3 responses are tool_use with no text blocks
        def _tool_only(call_id):
            resp = MagicMock()
            resp.stop_reason = "tool_use"
            resp.content = [
                ToolUseBlock(id=call_id, name="search", input={"query": "x"}),
            ]
            return resp

        mock_anthropic_client.messages.create.side_effect = [
            _tool_only("call_1"),
            _tool_only("call_2"),
            _tool_only("call_3"),  # final response after 2 rounds — no text
        ]
        gen = _make_generator(mock_anthropic_client)
        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "r"

        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        assert "try rephrasing" in result
