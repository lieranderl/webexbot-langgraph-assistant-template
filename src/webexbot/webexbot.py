import os
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot  # type: ignore
from .commands import OpenAI
from graph_reactagent.graph import create_default_graph
from graph_reactagent.invoker import GraphInvoker


def create_bot():
    load_dotenv()

    graph = create_default_graph()
    invoker = GraphInvoker(graph)
    openai_command = OpenAI(invoker)

    return WebexBot(
        teams_bot_token=os.getenv("WEBEX_TEAMS_ACCESS_TOKEN"),
        approved_domains=[os.getenv("WEBEX_TEAMS_DOMAIN")],
        bot_name="AI-Assistant",
        bot_help_subtitle="",
        threads=False,
        help_command=openai_command,
    )


if __name__ == "__main__":
    bot = create_bot()
    bot.run()
