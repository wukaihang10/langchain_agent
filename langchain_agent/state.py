from typing import NotRequired, TypedDict

from langchain.agents.middleware import AgentState


class FileEdition(TypedDict):
    file_path: str
    change_type: str
    old_path: NotRequired[str]


class GitAuditState(AgentState):
    edited_file_list: NotRequired[list[str]]
    edition_list: NotRequired[list[FileEdition]]
