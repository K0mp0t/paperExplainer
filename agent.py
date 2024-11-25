from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.tools import BaseTool

from typing import Annotated, Sequence,TypedDict

import json


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel
    tools: Sequence[BaseTool]


def has_tool_calls(message: BaseMessage) -> bool:
    try:
        _ = json.loads(message.content)
        return True
    except ValueError:
        return False


def tool_node(state: AgentState) -> dict[str: str]:
    tools_by_name = {tool.name: tool for tool in state["tools"]}
    outputs = []

    last_message = state["messages"][-1]
    if has_tool_calls(last_message):
        tool_calls = json.loads(last_message.content)
        for tool_call in tool_calls:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["parameters"])
            outputs.append(tool_result)

    return {"messages": outputs}


def agent_node(state: AgentState) -> dict[str: BaseMessage]:
    model_answer = state["model"].invoke(input=state["messages"])

    return {"messages": [model_answer]}
    

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if has_tool_calls(last_message):
        return "continue"
    else:
        return "end"