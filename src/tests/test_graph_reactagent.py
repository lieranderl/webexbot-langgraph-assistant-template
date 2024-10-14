import os
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from uuid import UUID
from graph_reactagent.messages_filter import DefaultMessageFilter  # type: ignore
from graph_reactagent.prompt_formatter import DefaultPromptFormatter  # type: ignore
from graph_reactagent.graph import Graph, create_default_graph  # type: ignore
from graph_reactagent.invoker import GraphInvoker  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage


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
        self.mock_graph = Mock(spec=Graph)
        self.mock_graph.graph = Mock()
        self.graph_invoker = GraphInvoker(
            graph=self.mock_graph,
            connection_str="sqlite:///test.db",
            llm_model="test-model",
            temperature=0.1,
        )

    def test_graph_invoker_initialization(self):
        mock_default_graph = Mock(spec=Graph)
        with patch(
            "graph_reactagent.graph.create_default_graph",
            return_value=mock_default_graph,
        ):
            invoker = GraphInvoker(graph=mock_default_graph)
            self.assertEqual(invoker.connection_str, os.getenv("SQL_CONNECTION_STR"))
            self.assertEqual(invoker.llm_model, os.getenv("LLM_MODEL"))
            self.assertEqual(invoker.temperature, 0.1)

    def test_graph_invoker_custom_initialization(self):
        custom_graph = Mock(spec=Graph)
        invoker = GraphInvoker(
            graph=custom_graph,
            connection_str="custom_conn_str",
            llm_model="custom_model",
            temperature=0.5,
        )
        self.assertEqual(invoker.connection_str, "custom_conn_str")
        self.assertEqual(invoker.llm_model, "custom_model")
        self.assertEqual(invoker.temperature, 0.5)
        self.assertEqual(invoker.graph, custom_graph)

    @patch("graph_reactagent.invoker.SqliteSaver.from_conn_string")
    def test_invoke_success(self, mock_sqlite_saver):
        expected_result = {"response": "Test response"}
        self.graph_invoker.graph.graph.invoke.return_value = expected_result

        result = self.graph_invoker.invoke("Test message")

        self.assertEqual(result, expected_result)
        self.graph_invoker.graph.graph.invoke.assert_called_once()
        called_input = self.graph_invoker.graph.graph.invoke.call_args[1]["input"]
        self.assertIsInstance(called_input["messages"][0], HumanMessage)
        self.assertEqual(called_input["messages"][0].content, "Test message")

    @patch("graph_reactagent.invoker.SqliteSaver.from_conn_string")
    def test_invoke_with_custom_kwargs(self, mock_sqlite_saver):
        self.graph_invoker.invoke("Test message", custom_param="custom_value")

        config = self.graph_invoker.graph.graph.invoke.call_args[1]["config"]
        self.assertEqual(config["configurable"]["custom_param"], "custom_value")

    def test_invoke_no_connection_string(self):
        with patch.object(GraphInvoker, "__init__", return_value=None):
            invoker = GraphInvoker(graph=self.mock_graph)
            invoker.connection_str = ""
            invoker.llm_model = "test-model"

        with self.assertRaisesRegex(
            ValueError, "No database connection string provided"
        ):
            invoker.invoke("Test message")

    def test_invoke_no_llm_model(self):
        with patch.object(GraphInvoker, "__init__", return_value=None):
            invoker = GraphInvoker(graph=self.mock_graph)
            invoker.connection_str = "sqlite:///test.db"
            invoker.llm_model = ""

        with self.assertRaisesRegex(ValueError, "No LLM model provided"):
            invoker.invoke("Test message")

    @patch("graph_reactagent.invoker.uuid4")
    @patch("graph_reactagent.invoker.SqliteSaver.from_conn_string")
    def test_invoke_run_id(self, mock_sqlite_saver, mock_uuid4):
        mock_uuid = UUID("12345678-1234-5678-1234-567812345678")
        mock_uuid4.return_value = mock_uuid

        self.graph_invoker.invoke("Test message")

        config = self.graph_invoker.graph.graph.invoke.call_args[1]["config"]
        self.assertEqual(config["run_id"], mock_uuid)

    @patch("graph_reactagent.invoker.SqliteSaver.from_conn_string")
    def test_invoke_saver_context_manager(self, mock_sqlite_saver):
        self.graph_invoker.invoke("Test message")

        mock_sqlite_saver.assert_called_once_with(self.graph_invoker.connection_str)
        mock_sqlite_saver.return_value.__enter__.assert_called_once()
        mock_sqlite_saver.return_value.__exit__.assert_called_once()

    @patch("graph_reactagent.invoker.SqliteSaver.from_conn_string")
    def test_invoke_sets_checkpointer(self, mock_sqlite_saver):
        self.graph_invoker.invoke("Test message")

        self.assertEqual(
            self.graph_invoker.graph.graph.checkpointer,
            mock_sqlite_saver.return_value.__enter__.return_value,
        )
