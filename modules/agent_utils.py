from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.tools import BaseTool

from arxiv import Search, Client
import requests
from requests_futures.sessions import FuturesSession

from pydantic import BaseModel, Field
from typing import Annotated, Sequence, TypedDict, Type, Any
from abc import ABC

import json

from modules.pdf_utils import process_pdf
from modules.utils import img2b64


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel
    tools: Sequence[BaseTool]

class ExtendedArxivRetrieverInput(BaseModel):
    arxiv_id: str = Field(description="arxiv paper id")


class ExtendedArxivRetriever(BaseTool, ABC):
    name: str = "extended_arxiv_retriever"
    description: str = """Extended arxiv tool: it can retrieve full text, images and tables from arxiv papers with 
        the arxiv id. Use only when you need extensive information about certain paper. Input should be a valid arxiv paper id."""
    return_direct: bool = True
    args_schema: Type[BaseModel] = ExtendedArxivRetrieverInput

    def _run(self, arxiv_id: str) -> list[dict[Any, Any]]:
        """Use the tool."""
        query = 'id:' + arxiv_id

        # arxiv ids are unique so we can just use first result
        result = next(Client().results(Search(query=query, max_results=1)), None)

        if result is None:
            return [{'type': 'text', 'text': 'Paper not found, perhaps the arxiv id is incorrect.'}]

        file = requests.get(result.pdf_url).content
        parsed_pdf = process_pdf(stream=file)

        # TODO: concatenate images?

        result = [{'type': 'text', 'text': parsed_pdf['text']}]
        result.extend([{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img2b64(img)}}
                       for img in parsed_pdf['images']])

        return result

    async def _arun(self, arxiv_id: str) -> str:
        """Use the tool asynchronously."""

        query = 'id:' + arxiv_id

        result = next(Client().results(Search(query=query, max_results=1)), None)

        if result is None:
            return 'Paper not found, perhaps the arxiv id is incorrect.'

        future = FuturesSession().get(result.pdf_url)
        file = future.result().content
        parsed_pdf = process_pdf(stream=file)

        # TODO: concatenate images?

        result = [{'type': 'text', 'text': parsed_pdf['text']}]
        result.extend([{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img2b64(img)}}
                       for img in parsed_pdf['images']])

        return result


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
            if isinstance(tool_result, list):
                outputs.append(HumanMessage(content=tool_result))
            elif isinstance(tool_result, str):
                outputs.append(f'Called tool {tool_call["name"]} with parameters {tool_call["parameters"]} and got {tool_result}')
            else:
                raise ValueError(f'Got unexpected result from tool {tool_call["name"]}: {tool_result}')

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