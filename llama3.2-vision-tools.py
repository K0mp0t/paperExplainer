from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

import json
import os

with open("./config.json") as f:
    config = json.load(f)

os.environ['TAVILY_API_KEY'] = config['tavily_api_key']

model = ChatOllama(model='llama3.2-vision')
tavily_search_tool = TavilySearchResults(max_results=5)

tool_description = """
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "A search engine optimized for comprehensive, accurate, and trusted results. 
                    Useful for when you need to answer questions about current events. Input should be a search query. 
                    This returns only the answer - not the original source data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "query": "search_query",
                            "type": "string"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        }
    ]
"""

system_prompt = SystemMessage(
    'You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along '
    'with the parameters required to run the tool. Respond as JSON using the following schema: '
    '{"tool_name": "function name", "parameters": {"parameterName": "parameterValue"}}. The tools are: ' + tool_description
)

human_prompt = HumanMessage(content="Find name of current Russia president and their height")

result = json.loads(model.invoke(input=[system_prompt, human_prompt]).content)
print('Tool params:', result)

result = tavily_search_tool.invoke(input=result['parameters'])
print('Tool output:', result)

tool_message = SystemMessage(content=json.dumps(result))

result = model.invoke(input=[tool_message, human_prompt])
print('Final answer:', result.content)
