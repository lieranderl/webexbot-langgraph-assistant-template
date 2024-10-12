## webex bot commands

# commnads that accept all text and send it back in replay
from webex_bot.models.command import Command  # type: ignore
from graph_reactagent.invoker import graph_db_invoke


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
        response = graph_db_invoke(
            message,
            thread_id=activity["target"]["globalId"],
            email=activity["actor"]["id"],
            displayName=activity["actor"]["displayName"],
            max_results=5,
        )

        return response["messages"][-1].content
