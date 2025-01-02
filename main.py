from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search.tool import TavilySearchResults

import json
import os

from modules.agent_utils import (ExtendedArxivRetriever, ExtendedArxivQueryRun, QnATool, build_graph)
from modules.utils import print_stream

with open("./config.json") as f:
    config = json.load(f)

os.environ['TAVILY_API_KEY'] = config['tavily_api_key']

model = ChatOllama(model='qwen2.5:14b', num_ctx=32768)
tools = [ExtendedArxivQueryRun(), TavilySearchResults(max_results=5), ExtendedArxivRetriever(), QnATool(model=model)]
model = model.bind_tools(tools)

system_message = config['system_message']

inputs = {"messages": [
    ("system", system_message),
    ("user", "Get first five references for paper titled Exploring Human-like Attention Supervision in Visual Question Answering. You can get the full text with tools")
],
    "model": model,
    "tools": tools}

graph = build_graph()

print_stream(graph.stream(inputs, stream_mode="values"))
