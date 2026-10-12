import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import build_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


if __name__ == "__main__":
    agent = build_agent()
    config: RunnableConfig = {"configurable": {"thread_id": "test_1"}}

    resp = agent.invoke(
        {"messages": [HumanMessage(content="我冰箱里有鸡胸肉、西兰花、番茄，帮我规划晚餐")]},
        config=config,
    )
    print(resp["messages"][-1].content)