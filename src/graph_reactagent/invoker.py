import os
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .graph import graph


async def graph_db_invoke(message, **kwargs):
    async with AsyncSqliteSaver.from_conn_string(
        os.getenv("SQL_CONNECTION_STR")
    ) as saver:
        graph.checkpointer = saver
        run_id = uuid4()
        return await graph.ainvoke(
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
