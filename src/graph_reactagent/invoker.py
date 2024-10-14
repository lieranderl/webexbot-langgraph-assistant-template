import os
from uuid import uuid4
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .interfaces import IGraphInvoker
from .graph import Graph


class GraphInvoker(IGraphInvoker):
    def __init__(self, graph: Graph, connection_str: str = ""):
        self.graph = graph
        self.connection_str = (
            connection_str or os.getenv("SQL_CONNECTION_STR") or "checkpoints.db"
        )

    def invoke(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        with SqliteSaver.from_conn_string(self.connection_str) as saver:
            self.graph.graph.checkpointer = saver
            run_id = uuid4()
            config = RunnableConfig(
                configurable={
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    **kwargs,
                },
                run_id=run_id,
            )
            result = self.graph.graph.invoke(
                input={"messages": [HumanMessage(content=message)]},
                config=config,
            )
            return result
