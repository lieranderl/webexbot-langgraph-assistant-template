from webex_bot.models.command import Command  # type: ignore
from graph_reactagent.interfaces import IGraphInvoker


class OpenAI(Command):
    def __init__(self, invoker: IGraphInvoker):
        super().__init__()
        self.invoker = invoker

    def execute(self, message, attachment_actions, activity):
        response = self.invoker.invoke(
            message,
            thread_id=activity["target"]["globalId"],
            email=activity["actor"]["id"],
            displayName=activity["actor"]["displayName"],
            max_results=5,
        )
        return response["messages"][-1].content
