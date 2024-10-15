from typing import List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage


class DefaultMessageFilter:
    def filter_messages(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        message_count = min(6, len(messages))
        filtered_messages = messages[-message_count:]

        while message_count > 1:
            if isinstance(filtered_messages[0], HumanMessage):
                break
            message_count -= 1
            while message_count > 0 and isinstance(
                messages[-message_count], (AIMessage, ToolMessage)
            ):
                # we cannot have an assistant message at the start of the chat history
                # if after removal of the first, we have an assistant message,
                # we need to remove the assistant message too
                # all tool messages should be preceded by an assistant message
                # if we remove a tool message, we need to remove the assistant message too
                message_count -= 1
            filtered_messages = messages[-message_count:]
        return filtered_messages
