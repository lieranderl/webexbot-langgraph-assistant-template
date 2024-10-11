import uuid
from dotenv import load_dotenv
from react_agent import graph

import asyncio

from react_agent.utils import get_message_text

thread_id = str(uuid.uuid4())

user_inputs = [
    "Hi there! My name is Will.",
    "Remember my name?",
    "What is 6 power 2?",
]


async def main():
    load_dotenv()
    # create config for the graph
    config = {"configurable": {"thread_id": thread_id, "temperature": 0.1}}
    # save the graph image to a file in src/static
    graph.get_graph().draw_mermaid_png(output_file_path="static/agent_graph.png")

    # run the graph
    cached_response_index = 0
    while True:
        try:
            user = input("User (q/Q to quit): ")
        except EOFError:
            user = user_inputs[cached_response_index]
            cached_response_index += 1
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        events = None
        events = graph.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user,
                    }
                ],
            },
            config=config,
            stream_mode="updates",
        )
        if user:
            async for event in events:
                if "call_model" in event:
                    response = get_message_text(event["call_model"]["messages"][-1])
                    if response:
                        print("AI:", response)

        if event and "prompt" in event:
            print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
