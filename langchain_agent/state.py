from typing import NotRequired, TypedDict, Literal

from langgraph.graph import MessagesState
from langchain.agents.middleware.todo import Todo
from langchain.messages import AnyMessage


class FileEdition(TypedDict):
    file_path: str
    change_type: str
    old_path: NotRequired[str]


class AgentState(MessagesState):
    todos: NotRequired[list[Todo]]
    edited_file_list: NotRequired[list[str]]
    edition_list: NotRequired[list[FileEdition]]
