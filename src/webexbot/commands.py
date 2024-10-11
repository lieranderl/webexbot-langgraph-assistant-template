## webex bot commands

# commnads that accept all text and send it back in replay
import asyncio
import logging
from webexteamssdk import WebexTeamsAPI
from webex_bot.models.command import Command
from react_agent import graph
from .invoker import graph_db_invoke

log = logging.getLogger(__name__)
webex_api = WebexTeamsAPI()


class OpenAI(Command):
    def __init__(self):
        super().__init__()

    # def pre_execute(self, message, attachment_actions, activity):
    #     """
    #     (optional function).
    #     Reply before running the execute function.

    #     Useful to indicate the bot is handling it if it is a long running task.

    #     :return: a string or Response object (or a list of either). Use Response if you want to return another card.
    #     """
    #     return "Working on it..."

    def execute(self, message, attachment_actions, activity):
        print("ACTIVITY:", activity)
        r = asyncio.run(graph_db_invoke(graph, activity["target"]["globalId"], activity["actor"]["id"], message))
        return r["messages"][-1].content
