from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END

import json
import os

from modules.agent_utils import AgentState, agent_node, tool_node, should_continue, ExtendedArxivRetriever, ExtendedArxivQueryRun

def print_stream(stream):
    idx = 0
    for s in stream:
        for message in s["messages"][idx:]:
            idx += 1
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

with open("./config.json") as f:
    config = json.load(f)

os.environ['TAVILY_API_KEY'] = config['tavily_api_key']

model = ChatOllama(model='qwen2.5:14b', num_ctx=32768)
tools = [ExtendedArxivQueryRun(), TavilySearchResults(max_results=5), ExtendedArxivRetriever()]
model = model.bind_tools(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tools_node", tool_node)

workflow.set_entry_point("agent_node")

workflow.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        "continue": "tools_node",
        "end": END,
    },
)

workflow.add_edge("tools_node", "agent_node")

graph = workflow.compile()

inputs = {"messages": [("user", "Get number of references for paper titled Exploring Human-like Attention Supervision in Visual Question Answering.")],
          "model": model,
          "tools": tools}
print_stream(graph.stream(inputs, stream_mode="values"))
