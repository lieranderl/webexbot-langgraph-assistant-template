import unittest
from unittest.mock import Mock, patch
from webexbot.commands import OpenAI  # type: ignore
from webexbot.webexbot import create_bot  # type: ignore


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.invoker = Mock()
        self.command = OpenAI(self.invoker)

    def test_execute(self):
        message = "Test message"
        attachment_actions = None
        activity = {
            "target": {"globalId": "test_thread"},
            "actor": {"id": "test_email", "displayName": "Test User"},
        }
        self.invoker.invoke.return_value = {"messages": [Mock(content="Test response")]}

        result = self.command.execute(message, attachment_actions, activity)

        self.invoker.invoke.assert_called_once_with(
            message,
            thread_id="test_thread",
            email="test_email",
            displayName="Test User",
            max_results=5,
        )
        self.assertEqual(result, "Test response")


@patch("webexbot.webexbot.WebexBot")
@patch("webexbot.webexbot.create_default_graph")
@patch("webexbot.webexbot.GraphInvoker")
@patch("webexbot.webexbot.OpenAI")
def test_create_bot(
    mock_openai, mock_graph_invoker, mock_create_default_graph, mock_webex_bot
):
    with patch.dict(
        "os.environ",
        {"WEBEX_TEAMS_ACCESS_TOKEN": "test_token", "WEBEX_TEAMS_DOMAIN": "test_domain"},
    ):
        bot = create_bot()

    mock_create_default_graph.assert_called_once()
    mock_graph_invoker.assert_called_once()
    mock_openai.assert_called_once()
    mock_webex_bot.assert_called_once()
    assert isinstance(
        bot, Mock
    )  # WebexBot is mocked, so the return value is a Mock object
