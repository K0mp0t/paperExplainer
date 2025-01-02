import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search.tool import TavilySearchResults

import json
import os

from modules.agent_utils import (ExtendedArxivRetriever, ExtendedArxivQueryRun, QnATool, build_graph)

with open("./config.json") as f:
    config = json.load(f)

os.environ['TAVILY_API_KEY'] = config['tavily_api_key']

model = ChatOllama(model='qwen2.5:14b', num_ctx=32768)
tools = [ExtendedArxivQueryRun(), TavilySearchResults(max_results=5), ExtendedArxivRetriever(), QnATool(model=model)]
model = model.bind_tools(tools)

system_message = config['system_message']

st.title("Research assistant")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_message}]

printed_msgs = len(st.session_state.messages)

for msg in st.session_state.messages:
    if msg["role"] == "system":
        st.markdown(msg["content"])

graph = build_graph()

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        printed_msgs += 1
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = {"messages": [(m["role"], m["content"]) for m in st.session_state.messages], "model": model,
              "tools": tools}

    response = graph.stream(inputs, stream_mode="values")
    for messages in response:
        for message in messages["messages"][printed_msgs:]:
            if len(message.content) > 0:
                message_content = message.content
            elif hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
                message_content = ''.join([f'Called tool {tc["name"]} with args: {tc["args"]}' for tc in message.tool_calls])
            else:
                continue
            with st.chat_message("assistant"):
                st.markdown(message_content[:1000] + ' <TRUNCATED>' if len(message_content) > 1000 else message_content)
            st.session_state.messages.append({"role": "assistant", "content": message.content})
            printed_msgs += 1

# Get first five references for paper titled Exploring Human-like Attention Supervision in Visual Question Answering. You can get the full text with tools
