from typing import Protocol, Any, Dict, List
from datetime import datetime
from langchain_core.messages import AnyMessage


class IMessageFilter(Protocol):
    def filter_messages(self, messages: List[AnyMessage]) -> List[AnyMessage]: ...


class IPromptFormatter(Protocol):
    def format_prompt(
        self, messages: List[AnyMessage], system_time: datetime
    ) -> Any: ...


class IGraphInvoker(Protocol):
    def invoke(self, message: str, **kwargs: Any) -> Dict[str, Any]: ...
