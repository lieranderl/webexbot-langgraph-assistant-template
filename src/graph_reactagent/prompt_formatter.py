from typing import Any, List
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from .interfaces import IPromptFormatter


class DefaultPromptFormatter(IPromptFormatter):
    def __init__(self, prompt: ChatPromptTemplate):
        self.prompt = prompt

    def format_prompt(self, messages: List[Any], system_time: datetime) -> Any:
        return self.prompt.invoke(
            {
                "messages": messages,
                "system_time": system_time.isoformat(),
            }
        )
