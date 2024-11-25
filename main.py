from langchain_ollama import ChatOllama
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END

import json
import os

from agent import AgentState, agent_node, tool_node, should_continue

with open("./config.json") as f:
    config = json.load(f)


with open("./tools.json") as f:
    tools_description = f.read()

os.environ['TAVILY_API_KEY'] = config['tavily_api_key']

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


model = ChatOllama(model='llama3.2-vision')
tools = [ArxivQueryRun(), TavilySearchResults(max_results=5)]

system_prompt = '''You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along '
        with the parameters required to run the tool. You are alowed to execute multiple tools.
        Respond only with JSON using the following schema: 
        [{"name": "function name", "parameters": {"parameterName": "parameterValue"}}]. 
        If you decide that no tool is needed at the moment return final answer.
        The tools are: ''' + tools_description

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

inputs = {"messages": [("system", system_prompt),
                       ("user", "get me title of the paper with arxiv id 1706.03762 and its publication date")],
          "model": model,
          "tools": tools}
print_stream(graph.stream(inputs, stream_mode="values"))


