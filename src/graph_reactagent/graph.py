from typing import Sequence
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.messages import BaseMessage, ToolMessage
from .tools import search, get_webex_user_info, power

# Create a ToolNode with the provided tools
tools = ToolNode([search, get_webex_user_info, power])

# Initialize the OpenAI model, specifying the model type
model = ChatOpenAI(model="gpt-4o")


# Define a custom messages modifier to reduce token usage and avoid sending all messages to the LLM.
# This function should take in a list of messages and return a filtered list of messages.
def messages_modifier(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """
    Custom messages modifier function to reduce token usage.

    Args:
        messages (Sequence[BaseMessage]): The list of messages to be modified.

    Returns:
        Sequence[BaseMessage]: The modified list of messages.
    """
    # Filter messages to not interrupt tool calls. ToolMessage must be sent with the preceding message of a tool call.
    if isinstance(messages[-1], ToolMessage):
        return messages[-2:]
    return messages[-1:]


# Create the ReAct agent using the model, tools, and the custom messages modifier
graph = create_react_agent(model, tools, messages_modifier=messages_modifier)

# Set the name of the ReAct agent to track LangSmith
graph.name = "Webex Bot Demo ReAct Agent"
