from typing import Protocol, Any, Dict, List
from datetime import datetime


class IMessageFilter(Protocol):
    def filter_messages(self, messages: List[Any]) -> List[Any]: ...


class IPromptFormatter(Protocol):
    def format_prompt(self, messages: List[Any], system_time: datetime) -> Any: ...


class IGraphInvoker(Protocol):
    def invoke(self, message: str, **kwargs: Any) -> Dict[str, Any]: ...
