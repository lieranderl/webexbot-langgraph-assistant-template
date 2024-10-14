import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from graph_reactagent.messages_filter import DefaultMessageFilter  # type: ignore
from graph_reactagent.prompt_formatter import DefaultPromptFormatter  # type: ignore
from graph_reactagent.graph import Graph, create_default_graph  # type: ignore
from graph_reactagent.invoker import GraphInvoker  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_core.tools import Tool


class TestDefaultMessageFilter(unittest.TestCase):
    def test_filter_messages(self):
        filter = DefaultMessageFilter()
        messages = [Mock() for _ in range(10)]
        filtered = filter.filter_messages(messages)
        self.assertEqual(len(filtered), 5)


class TestDefaultPromptFormatter(unittest.TestCase):
    def setUp(self):
        self.prompt = Mock(spec=ChatPromptTemplate)
        self.formatter = DefaultPromptFormatter(self.prompt)

    def test_format_prompt(self):
        messages = [Mock() for _ in range(3)]
        system_time = datetime.now(timezone.utc)
        self.formatter.format_prompt(messages, system_time)
        self.prompt.invoke.assert_called_once()


# Dummy tool functions for testing
def dummy_search(query: str) -> str:
    return f"Search results for: {query}"


def dummy_power(a: int, b: int) -> int:
    return a**b


def dummy_get_webex_user_info() -> dict:
    return {"displayName": "Test User", "email": "test@example.com"}


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-3.5-turbo" 
        self.tools = [
            Tool.from_function(
                func=dummy_search, name="search", description="Search the web"
            ),
            Tool.from_function(
                func=dummy_power, name="power", description="Calculate power"
            ),
            Tool.from_function(
                func=dummy_get_webex_user_info,
                name="get_webex_user_info",
                description="Get Webex user info",
            ),
        ]
        self.message_filter = Mock(spec=DefaultMessageFilter)
        self.prompt_formatter = Mock(spec=DefaultPromptFormatter)
        self.graph = Graph(
            self.model, self.tools, self.message_filter, self.prompt_formatter
        )

    def test_format_for_model(self):
        state = {"messages": [Mock() for _ in range(5)]}
        self.graph._format_for_model(state)
        self.message_filter.filter_messages.assert_called_once()
        self.prompt_formatter.format_prompt.assert_called_once()


@patch("graph_reactagent.graph.ChatOpenAI")
@patch("graph_reactagent.graph.ToolNode")
@patch("graph_reactagent.graph.create_react_agent")
def test_create_default_graph(
    mock_create_react_agent, mock_tool_node, mock_chat_openai
):
    create_default_graph()
    mock_chat_openai.assert_called_once_with(model="gpt-4o")
    mock_tool_node.assert_called_once()
    mock_create_react_agent.assert_called_once()


class TestGraphInvoker(unittest.TestCase):
    def setUp(self):
        self.graph = Mock()
        self.graph.graph = Mock()
        self.graph.graph.invoke.return_value = {
            "messages": [Mock(content="Test response")]
        }
        self.invoker = GraphInvoker(self.graph)

    @patch("graph_reactagent.invoker.SqliteSaver")
    def test_invoke(self, mock_sqlite_saver):
        message = "Test message"
        result = self.invoker.invoke(message)

        # Assert that SqliteSaver.from_conn_string was called
        mock_sqlite_saver.from_conn_string.assert_called_once()

        # Assert that the graph's invoke method was called with the correct arguments
        self.graph.graph.invoke.assert_called_once()
        call_args = self.graph.graph.invoke.call_args
        self.assertIn("input", call_args[1])
        self.assertIn("config", call_args[1])

        # Assert that the result is a dictionary and has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertEqual(result["messages"][0].content, "Test response")
