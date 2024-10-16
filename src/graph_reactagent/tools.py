from typing import Annotated, Any, Optional, List, Dict, cast
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_community.tools.tavily_search.tool import TavilySearchResults


def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Dict[str, Any]]]:
    """Search for general web results.
       Search for real-time information using the Tavily search engine.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    max_results: int = config.get("configurable", {}).get("max_results") or 5
    wrapped = TavilySearchResults(max_results=max_results)
    result = wrapped.invoke({"query": query})
    return cast(List[Dict[str, Any]], result)


def power(a: int, b: int) -> int:
    """Calculate power of number.
    input: a, b
    """
    return a**b


def get_webex_user_info(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> Optional[Dict[str, str]]:
    """Get user information: email and user name/displayName from Webex SDK"""
    displayName: str = config.get("configurable", {}).get("displayName") or ""
    email: str = config.get("configurable", {}).get("email") or ""
    return {"displayName": displayName, "email": email}
