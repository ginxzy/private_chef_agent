import os
from langchain_tavily import TavilySearch


search_tool = TavilySearch(
    max_results=2,
    include_domains=["xiachufang.com", "douguo.com"],
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


def get_tools():
    """返回 Agent 需要的所有工具"""
    return [search_tool]