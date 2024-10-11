from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def graph_db_invoke(graph, thread_id, message, **kwargs):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph.checkpointer = saver
        run_id = uuid4()
        print(kwargs)
        return await graph.ainvoke(
            input={"messages": HumanMessage(content=message)},
            config=RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    "model": "openai/gpt-4o-mini",
                    "temperature": 0.1,
                    **kwargs,
                },
                run_id=run_id,
            ),
        )
