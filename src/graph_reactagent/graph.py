from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .tools import search
from .tools import get_webex_user_info
from .tools import power

tools = ToolNode([search, get_webex_user_info, power])
model = ChatOpenAI(model="gpt-4o")
graph = create_react_agent(model, tools, checkpointer=MemorySaver())
graph.name = "Webex Bot Demo ReAct Agent"

