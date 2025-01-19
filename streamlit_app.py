import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults

import json

from modules.agent_utils import (ExtendedArxivRetriever, ExtendedArxivQueryRun, build_graph, GitHubRepoStructureViewer,
                                 GitHubFileViewer)

with open("./config.json") as f:
    config = json.load(f)

model = ChatOllama(model=config["model"], num_ctx=config["num_ctx"])
tools = [ExtendedArxivQueryRun(), DuckDuckGoSearchResults(), ExtendedArxivRetriever(),
         GitHubRepoStructureViewer(github_app_id=config["github_app_id"],
                                   github_app_private_key=config["github_app_private_key"]),
         GitHubFileViewer(github_app_id=config["github_app_id"],
                          github_app_private_key=config["github_app_private_key"])]
model = model.bind_tools(tools)

st.title("Research assistant")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": config['system_message'], "raw_msg": SystemMessage(config['system_message'])}]

printed_msgs = len(st.session_state.messages) - 1

for msg in st.session_state.messages:
    if msg["role"] == "system":
        with st.chat_message("system"):
            st.markdown(msg["content"])

graph = build_graph()
if not hasattr(st.session_state, "num_tokens"):
    st.session_state.num_tokens = 0
st.metric(label="Number of tokens", value=st.session_state.num_tokens)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        printed_msgs += 1
    st.session_state.messages.append({"role": "user", "content": prompt, "raw_msg": HumanMessage(prompt)})

    inputs = {"messages": [msg["raw_msg"] for msg in st.session_state.messages if msg["role"] != "system"],
              "model": model, "tools": tools, "system_message": SystemMessage(config['system_message'])}

    response = graph.stream(inputs, stream_mode="values")
    for messages in response:
        for message in messages["messages"][printed_msgs:]:
            if len(message.content) > 0:
                message_content = message.content
            elif hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
                message_content = ''.join([f'Called tool {tc["name"]} with args: {tc["args"]}' for tc in message.tool_calls])
            else:
                # continue
                message_content = ''
            if isinstance(message, AIMessage):
                role = 'assistant'
            elif isinstance(message, HumanMessage):
                role = 'user'
            elif isinstance(message, ToolMessage):
                role = 'tool'
            else:
                role = 'unknown'
            if hasattr(message, 'usage_metadata'):
                st.session_state.num_tokens = message.usage_metadata['total_tokens']
            with st.chat_message(role):
                if len(message_content) > config['max_response_length'] and role == 'tool':
                    message_content = message_content[:config['max_response_length']] + ' <TRUNCATED>'
                st.markdown(message_content)
            st.session_state.messages.append({"role": role, "content": message.content, "raw_msg": message})
            printed_msgs += 1

# Get first five references for paper titled Exploring Human-like Attention Supervision in Visual Question Answering. You can get the full text with tools
