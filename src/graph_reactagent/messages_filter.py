from typing import Any, List
from langchain_core.messages import ToolMessage
from .interfaces import IMessageFilter


class DefaultMessageFilter(IMessageFilter):
    def filter_messages(self, messages: List[Any]) -> List[Any]:
        messages_length = len(messages)

        # If there are fewer than 5 messages, return all messages
        if messages_length < 5:
            return messages

        # Start with the last 5 messages
        filtered_messages = messages[-5:]

        # If the first message in filtered_messages is a ToolMessage,
        # add one preceding message if available
        if isinstance(filtered_messages[0], ToolMessage) and messages_length > 5:
            filtered_messages = messages[-6:]

        return filtered_messages
