from datetime import datetime, timezone
from typing import Any
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from .tools import search, get_webex_user_info, power

# Create a ToolNode with the provided tools
tools = ToolNode([search, get_webex_user_info, power])

# Initialize the OpenAI model, specifying the model type
model = ChatOpenAI(model="gpt-4o")

# Define the system prompt template with the system time
SYSTEM_PROMPT = """You are a helpful AI assistant in a Webex chat. 
Your primary tasks include:
1. Searching the web to provide relevant information.
2. Retrieving personal information from Webex.
3. Performing power calculations.
4. Interacting with users to assist with their queries.
Ensure your responses are clear, accurate, and concise. Always aim to provide the most helpful information based on the user's request.
System time: {system_time}"""

# Create a chat prompt template from the system prompt and message placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]
)


def format_for_model(state) -> Any:
    # Limit the number of messages sent to the model to the last 5, reducing token usage.
    # Ensure tool-related messages are not interrupted by including the message immediately before a tool call along with the ToolMessage.
    messages = state["messages"]
    messages_length = len(messages)
    # If the length of messages is less than or equal to 5, handle accordingly
    if messages_length <= 5:
        # Check if the last message is an instance of ToolMessage
        if isinstance(messages[-1], ToolMessage) and messages_length > 1:
            messages = messages[:-1]
    else:
        # Get the last 5 messages
        messages = messages[-5:]
        # Check if the first of these last 5 messages is a ToolMessage, and if so, include one more preceding message
        if isinstance(messages[0], ToolMessage) and messages_length > 5:
            messages = state["messages"][-6:]

    return prompt.invoke(
        {
            "messages": messages,
            "system_time": datetime.now(timezone.utc).astimezone().isoformat(),
        }
    )


# Create the ReAct agent using the model, tools, and the custom state modifier
graph = create_react_agent(model, tools, state_modifier=format_for_model)

# Set the name of the ReAct agent to track LangSmith
graph.name = "Webex Bot Demo ReAct Agent"
