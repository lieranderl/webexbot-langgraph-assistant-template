# tests/test_message_filter.py

import unittest
from typing import List, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph_reactagent.messages_filter import DefaultMessageFilter  # type: ignore


class TestDefaultMessageFilter(unittest.TestCase):
    def setUp(self):
        self.filter = DefaultMessageFilter()

    def test_empty_message_list(self):
        messages: List[Any] = []
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_less_than_five_messages(self):
        messages = [HumanMessage(content=str(i)) for i in range(4)]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_exactly_five_messages(self):
        messages = [HumanMessage(content=str(i)) for i in range(5)]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages)

    def test_more_than_five_messages(self):
        messages = [HumanMessage(content=str(i)) for i in range(10)]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages[-5:])

    def test_first_of_last_five_is_tool_message(self):
        messages = [HumanMessage(content=str(i)) for i in range(5)]
        messages.append(ToolMessage(content="tool", tool_call_id="tool1"))
        messages.extend([HumanMessage(content=str(i)) for i in range(6, 10)])
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages[-6:])

    def test_first_of_last_five_is_not_tool_message(self):
        messages = [HumanMessage(content=str(i)) for i in range(5)]
        messages.append(ToolMessage(content="tool", tool_call_id="tool1"))
        messages.extend([HumanMessage(content=str(i)) for i in range(6, 11)])
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages[-5:])

    def test_mixed_message_types(self):
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            ToolMessage(content="5", tool_call_id="tool2"),
            HumanMessage(content="6"),
        ]
        filtered = self.filter.filter_messages(messages)
        self.assertEqual(filtered, messages[-5:])


if __name__ == "__main__":
    unittest.main()
