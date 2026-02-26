import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools:

1. **search_course_content** — Search within course lesson text for specific topics, concepts, or details.
2. **get_course_outline** — Retrieve a course's title, link, and full lesson list. Use this when asked about what a course covers, its outline, structure, overview, or lesson list.

Tool Usage:
- Use `get_course_outline` for questions about course structure, outlines, overviews, or "what lessons are in" a course
- Use `search_course_content` for questions about specific topics, concepts, or details within course content
- **Up to two tool calls per query** — you may call one tool, review the results, then call a second tool if needed
- Example workflow: use `get_course_outline` to find the course structure, then `search_course_content` to look up details from a specific lesson
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course outline/overview questions**: Use `get_course_outline`, then present the course title, course link, and complete lesson list (number + title for each)
- **Course content questions**: Use `search_course_content`, then synthesize results into an accurate response
- **Multi-step questions**: If a question requires information from multiple sources (e.g., course structure AND specific content), use tools sequentially to gather all needed information before responding
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters
        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Initial API call
        response = self.client.messages.create(**api_params)

        # Tool-calling loop: execute tools and follow up, up to MAX_TOOL_ROUNDS times
        for round_num in range(self.MAX_TOOL_ROUNDS):
            if response.stop_reason != "tool_use" or not tool_manager:
                break

            tool_results = self._execute_tool_calls(response, tool_manager)
            if not tool_results:
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            api_params["messages"] = messages

            response = self.client.messages.create(**api_params)

        # Extract text from the final response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return "I wasn't able to complete my analysis. Please try rephrasing your question."

    def _execute_tool_calls(self, response, tool_manager) -> list:
        """
        Execute all tool_use blocks in a response.

        Args:
            response: The API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dicts, empty if no tool_use blocks found
        """
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = tool_manager.execute_tool(block.name, **block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        return tool_results