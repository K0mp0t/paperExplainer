from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain.tools import BaseTool
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from arxiv import Search, Client
import requests
from requests_futures.sessions import FuturesSession
import logging
import re
from pydantic import BaseModel, Field
from typing import Annotated, Sequence, TypedDict, Type, Any
from abc import ABC
import json

from modules.pdf_utils import process_pdf


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel
    tools: Sequence[BaseTool]


class ExtendedArxivAPIWrapper(ArxivAPIWrapper):
    def run(self, query: str) -> str:
        logger = logging.getLogger(__name__)
        try:
            results = self._fetch_results(
                query
            )  # Using helper function to fetch results
        except self.arxiv_exceptions as ex:
            logger.error(f"Arxiv exception: {ex}")  # Added error logging
            return f"Arxiv exception: {ex}"
        docs = [
            f"Entry ID: {result.entry_id}\n"
            f"Published: {result.updated.date()}\n"
            f"Title: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"


class ExtendedArxivQueryRun(ArxivQueryRun):  # type: ignore[override, override]
    """Tool that searches the Arxiv API."""

    description: str = (
        "A wrapper around Arxiv.org "
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Input should be a search query. Outputs arxiv ID, published date, title, authors, and summary."
    )
    api_wrapper: ArxivAPIWrapper = Field(default_factory=ExtendedArxivAPIWrapper)


class ExtendedArxivRetrieverInput(BaseModel):
    arxiv_id: str = Field(description="arxiv paper id")


class ExtendedArxivRetriever(BaseTool, ABC):
    name: str = "extended_arxiv_retriever"
    description: str = (
        "Extended arxiv tool: it can retrieve full text including abstract, appendices and references with the arxiv id."
        "Before performing a search make sure you've found the right arxiv ID."
        "Use it only when you need full text for better paper understanding. Input must be a valid arxiv paper id."
    )
    return_direct: bool = True
    args_schema: Type[BaseModel] = ExtendedArxivRetrieverInput

    def is_arxiv_identifier(self, query: str) -> bool:
        arxiv_identifier_pattern = r"\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+|)|\d{7}.*"
        for query_item in query.split():
            match_result = re.match(arxiv_identifier_pattern, query_item)
            if not match_result:
                return False
            assert match_result is not None
            if not match_result.group(0) == query_item:
                return False
        return True

    def _run(self, arxiv_id: str) -> list[dict[Any, Any]]:
        """Use the tool."""
        if not self.is_arxiv_identifier(arxiv_id):
            return [{'type': 'text', 'text': 'Paper not found, perhaps the arxiv id is incorrect. '
                                             'Remember to remove preprint version and link attributes like http:// from arxiv id and try again'}]
        query = 'id:' + arxiv_id

        # arxiv ids are unique so we can just use first result
        result = next(Client().results(Search(query=query, max_results=1)), None)

        if result is None:
            return [{'type': 'text', 'text': 'Paper not found, perhaps the arxiv id is incorrect. '
                                             'Remember to remove preprint version and link attributes like http:// from arxiv id and try again'}]

        file = requests.get(result.pdf_url).content
        parsed_pdf = process_pdf(stream=file)

        return parsed_pdf['text']

    async def _arun(self, arxiv_id: str) -> list[dict[Any, Any]]:
        """Use the tool asynchronously."""

        query = 'id:' + arxiv_id

        result = next(Client().results(Search(query=query, max_results=1)), None)

        if result is None:
            return [{'type': 'text', 'text': 'Paper not found, perhaps the arxiv id is incorrect.'}]

        future = FuturesSession().get(result.pdf_url)
        file = future.result().content
        parsed_pdf = process_pdf(stream=file)

        return parsed_pdf['text']


class QnAToolInput(BaseModel):
    arxiv_id: str = Field(description="your question with context")


class QnATool(BaseTool, ABC):
    name: str = 'qna_tool'
    description: str = ('This tool allows you to ask LLM a question with large context (for example, a scientific paper '
                        'full text or its source code). The answer is returned as plain text. Highly useful if you need '
                        'to better understand some details of a paper.')
    model: BaseChatModel
    args_schema: Type[BaseModel] = QnAToolInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.model.invoke(input=query).content

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        return self.model.invoke(input=query).content


def tool_node(state: AgentState) -> dict[str: str]:
    tools_by_name = {tool.name: tool for tool in state["tools"]}
    outputs = []

    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(ToolMessage(
            content=json.dumps(tool_result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        ))

    return {"messages": outputs}


def agent_node(state: AgentState) -> dict[str: BaseMessage]:
    model_answer = state["model"].invoke(input=state["messages"])

    return {"messages": [model_answer]}
    

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if len(last_message.tool_calls) > 0:
        return "continue"
    else:
        return "end"