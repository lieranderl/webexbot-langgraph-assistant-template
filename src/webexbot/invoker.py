from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from app.routers.langgraph.models import ChatMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def graph_db_invoke(graph, thread_id, email, message):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph.checkpointer = saver
        run_id = uuid4()
        input_message = ChatMessage(type="human", content=message)
        return await graph.ainvoke(
            input={"messages": [input_message.to_langchain()]},
            config=RunnableConfig(
                configurable={
                    "thread_id": thread_id,
                    "model": "openai/gpt-4o-mini",
                    "temperature": 0.1,
                    "email": email,
                },
                run_id=run_id,
            )
        )
