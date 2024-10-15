import unittest
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from graph_reactagent.messages_filter import DefaultMessageFilter  # type: ignore


class TestDefaultMessageFilter(unittest.TestCase):
    def setUp(self):
        self.filter = DefaultMessageFilter()

    def test_empty_message_list(self):
        messages: List[AnyMessage] = []
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_less_than_six_messages(self):
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_exactly_six_messages(self):
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
            AIMessage(content="6"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_more_than_six_messages(self):
        messages = [HumanMessage(content=str(i)) for i in range(10)]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages[-6:])

    def test_no_assistant_or_tool_message_at_start(self):
        messages = [
            AIMessage(content="1"),
            ToolMessage(content="2", tool_call_id="tool1"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
            AIMessage(content="6"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(len(filtered), 4)
        self.assertIsInstance(filtered[0], HumanMessage)

    def test_tool_message_preceded_by_assistant(self):
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            ToolMessage(content="3", tool_call_id="tool1"),
            HumanMessage(content="4"),
            AIMessage(content="5"),
            ToolMessage(content="6", tool_call_id="tool2"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(len(filtered), 6)
        tool_indices = [
            i for i, msg in enumerate(filtered) if isinstance(msg, ToolMessage)
        ]
        for idx in tool_indices:
            self.assertIsInstance(filtered[idx - 1], AIMessage)

    def test_remove_orphaned_assistant_and_tool_messages(self):
        messages = [
            AIMessage(content="1"),
            ToolMessage(content="2", tool_call_id="tool1"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            ToolMessage(content="5", tool_call_id="tool2"),
            HumanMessage(content="6"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(len(filtered), 4)
        expected = [
            HumanMessage(content="3"),
            AIMessage(content="4"),
            ToolMessage(content="5", tool_call_id="tool2"),
            HumanMessage(content="6"),
        ]
        self.assertEqual(filtered, expected)

    def test_complex_scenario(self):
        messages = [
            AIMessage(content="0"),
            HumanMessage(content="1"),
            AIMessage(content="2"),
            ToolMessage(content="3", tool_call_id="tool1"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
            AIMessage(content="6"),
            ToolMessage(content="7", tool_call_id="tool2"),
            AIMessage(content="8"),
            HumanMessage(content="9"),
        ]
        filtered = self.filter.filter_messages(messages)
        expected = [
            HumanMessage(content="5"),
            AIMessage(content="6"),
            ToolMessage(content="7", tool_call_id="tool2"),
            AIMessage(content="8"),
            HumanMessage(content="9"),
        ]
        self.assertEqual(filtered, expected)


if __name__ == "__main__":
    unittest.main()
