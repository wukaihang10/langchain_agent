from typing import NotRequired, TypedDict, Literal

from langgraph.graph import MessagesState
from langchain.agents.middleware.todo import Todo
from langchain.messages import AnyMessage

from langchain_agent.permissions.types import ToolExecutionDecision


class FileEdition(TypedDict):
    file_path: str
    change_type: str
    old_path: NotRequired[str]


class AgentState(MessagesState):
    todos: NotRequired[list[Todo]]
    tool_decisions: NotRequired[dict[str, ToolExecutionDecision]]
    edited_file_list: NotRequired[list[str]]
    edition_list: NotRequired[list[FileEdition]]
