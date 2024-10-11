"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Annotated, Any, Callable, List, Optional, cast


from langchain_core.runnables import RunnableConfig

from react_agent.configuration import Configuration
from langchain_core.tools import InjectedToolArg
from langchain_community.tools.tavily_search.tool import TavilySearchResults


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.
       Search for real-time information using the Tavily search engine.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def power(a: int, b: int) -> int:
    """Calculate power of a number."""
    return a**b


async def get_webex_user_info(config: RunnableConfig) -> Optional[dict[str, Any]]:
    """Get user information: email and user name/display name from Webex SDK"""
    displayName = config.get("configurable", {}).get("displayName") or ""
    email = config.get("configurable", {}).get("email") or ""

    return {
        "displayName": displayName,
        "email": email,
    }


TOOLS: List[Callable[..., Any]] = [search, get_webex_user_info, power]
