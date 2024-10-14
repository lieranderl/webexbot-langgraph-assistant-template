from typing import Any, Dict
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.prompts import ChatPromptTemplate
from .interfaces import IMessageFilter, IPromptFormatter
from .messages_filter import DefaultMessageFilter
from .prompt_formatter import DefaultPromptFormatter
from .tools import search, get_webex_user_info, power

SYSTEM_PROMPT = """You are a helpful AI assistant in a Webex chat. 
Your primary tasks include:
1. Searching the web to provide relevant information.
2. Retrieving personal information from Webex.
3. Performing power calculations.
4. Interacting with users to assist with their queries.
Ensure your responses are clear, accurate, and concise. Always aim to provide the most helpful information based on the user's request.
System time: {system_time}"""


class Graph:
    def __init__(
        self,
        model: str,
        tools: list,
        message_filter: IMessageFilter,
        prompt_formatter: IPromptFormatter,
        name: str = "ReAct Agent",
    ):
        self.model = ChatOpenAI(model=model)
        self.tools = ToolNode(tools)
        self.message_filter = message_filter
        self.prompt_formatter = prompt_formatter
        self.name = name

        self.graph = self._create_graph()

    def _create_graph(self):
        graph = create_react_agent(
            self.model, self.tools, state_modifier=self._format_for_model
        )
        graph.name = self.name
        return graph

    def _format_for_model(self, state: Dict[str, Any]) -> Any:
        messages = state["messages"]
        filtered_messages = self.message_filter.filter_messages(messages)
        return self.prompt_formatter.format_prompt(
            filtered_messages, datetime.now(timezone.utc).astimezone()
        )


def create_default_graph():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )
    tools = [get_webex_user_info, search, power]
    return Graph(
        model="gpt-4o",
        tools=tools,
        message_filter=DefaultMessageFilter(),
        prompt_formatter=DefaultPromptFormatter(prompt),
        name="Webex Bot Demo ReAct Agent",
    )
