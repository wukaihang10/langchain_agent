from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph_agent.llm.client import chat

from langgraph_agent.ragservice.models import BuiltContext
from langgraph_agent.ragservice.prompt_builder import Message, RAGPromptBuilder

ChatFunction = Callable[..., Any]


class RAGGenerator:
    """
    使用现有的 LLM Client，
    根据问题和检索上下文生成答案。
    """

    def __init__(
        self,
        prompt_builder: RAGPromptBuilder | None = None,
        chat_function: ChatFunction = chat,
    ) -> None:
        self.prompt_builder = (
            prompt_builder if prompt_builder is not None else RAGPromptBuilder()
        )

        self.chat_function = chat_function

    def generate(
        self,
        question: str,
        context: BuiltContext,
    ) -> str:
        messages = self.prompt_builder.build_messages(
            question=question,
            context=context,
        )

        assistant_message = self.chat_function(messages=messages, tools=[])

        content = getattr(assistant_message, "content", None)

        if not isinstance(content, str):
            raise ValueError("LLM returned no text content")

        answer = content.strip()

        if not answer:
            raise ValueError("LLM returned an empty answer")

        return answer
