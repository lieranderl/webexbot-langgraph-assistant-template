import os
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .graph import graph


def graph_db_invoke(message, **kwargs):
    with SqliteSaver.from_conn_string(os.getenv("SQL_CONNECTION_STR")) as saver:
        graph.checkpointer = saver
        run_id = uuid4()
        return graph.invoke(
            input={"messages": HumanMessage(content=message)},
            config=RunnableConfig(
                configurable={
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    **kwargs,
                },
                run_id=run_id,
            ),
        )
