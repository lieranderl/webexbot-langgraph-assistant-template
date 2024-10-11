import os
from uuid import uuid4
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .graph import graph


def graph_db_invoke(message: str, **kwargs: Any) -> dict[str, Any] | Any:
    """
    Invokes the graph with a given message and additional keyword arguments, using a SQLite database for checkpointing.

    Args:
        message (str): The input message to be processed by the graph.
        **kwargs (Any): Additional configuration options for the graph.

    Returns:
        Any: The output from the graph invocation.
    """
    connection = os.getenv("SQL_CONNECTION_STR") or "checkpoints.db"
    with SqliteSaver.from_conn_string(connection) as saver:
        graph.checkpointer = saver
        run_id = uuid4()
        config = RunnableConfig(
            configurable={
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                **kwargs,
            },
            run_id=run_id,
        )
        return graph.invoke(
            input={"messages": HumanMessage(content=message)},
            config=config,
        )
