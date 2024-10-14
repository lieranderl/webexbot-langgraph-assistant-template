import os
from uuid import uuid4
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .interfaces import IGraphInvoker
from .graph import Graph


class GraphInvoker(IGraphInvoker):
    def __init__(
        self,
        graph: Graph,
        connection_str: str = "",
        llm_model: str = "",
        temperature: float = 0.1,
    ):
        self.graph = graph
        self.connection_str = connection_str or os.getenv("SQL_CONNECTION_STR")
        self.llm_model = llm_model or os.getenv("LLM_MODEL")
        self.temperature = temperature

    def invoke(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.connection_str:
            raise ValueError("No database connection string provided")
        if not self.llm_model:
            raise ValueError("No LLM model provided")
        with SqliteSaver.from_conn_string(self.connection_str) as saver:
            self.graph.graph.checkpointer = saver
            run_id = uuid4()
            config = RunnableConfig(
                configurable={
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    **kwargs,
                },
                run_id=run_id,
            )
            result = self.graph.graph.invoke(
                input={"messages": [HumanMessage(content=message)]},
                config=config,
            )
            return result
