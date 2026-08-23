from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from langchain_agent.llm.client import (
    LLMClientError,
    chat,
)

ChatFunction = Callable[..., Any]


class QueryExpansionError(RuntimeError):
    """
    Query expansion failed.

    Query expansion is an optional retrieval enhancement,
    so callers may choose to fall back to the original query.
    """


class QueryExpander(Protocol):
    """
    生成原始查询之外的补充检索表达。

    注意：
    expand() 不需要返回原始 query。
    原始 query 由 MultiQueryRetriever 强制保留。
    """

    def expand(
        self,
        query: str,
    ) -> list[str]: ...


class IdentityQueryExpander:
    """
    不生成任何额外 query 的 baseline。
    """

    def expand(
        self,
        query: str,
    ) -> list[str]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        return []


class LLMQueryExpander:
    """
    使用 LLM 将自然语言问题转换为更适合
    Python repository retrieval 的补充查询。

    主要解决：

        中文自然语言
            ↓
        英文代码术语 / identifier vocabulary

    原始 query 不在这里返回，
    由 MultiQueryRetriever 统一保留。
    """

    def __init__(
        self,
        *,
        max_rewrites: int = 2,
        chat_function: ChatFunction = chat,
    ) -> None:
        if max_rewrites <= 0:
            raise ValueError("max_rewrites must be " "greater than 0")

        self.max_rewrites = max_rewrites
        self.chat_function = chat_function

        # 仅进程内缓存。
        # Evaluation 同一个 query 会跑多个 Retriever，
        # 避免重复调用 LLM。
        self._cache: dict[
            str,
            tuple[str, ...],
        ] = {}

    def expand(
        self,
        query: str,
    ) -> list[str]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            return []

        cached = self._cache.get(normalized_query)

        if cached is not None:
            return list(cached)

        messages = self._build_messages(normalized_query)

        try:
            assistant_message = self.chat_function(
                messages=messages,
                tools=[],
            )

        except LLMClientError as error:
            raise QueryExpansionError("LLM query expansion failed") from error

        content = getattr(
            assistant_message,
            "content",
            None,
        )

        if not isinstance(content, str):
            raise QueryExpansionError("LLM query expansion returned " "no text content")

        rewrites = self._parse_rewrites(
            content=content,
            original_query=(normalized_query),
        )

        self._cache[normalized_query] = tuple(rewrites)

        return list(rewrites)

    def _build_messages(
        self,
        query: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You rewrite user questions into "
                    "search queries for a Python source "
                    "code repository.\n\n"
                    "Your goal is retrieval, not answering "
                    "the question.\n\n"
                    "Generate concise alternative queries "
                    "that help match English Python code, "
                    "identifiers, comments, and implementation "
                    "terminology.\n\n"
                    "Rules:\n"
                    "- Preserve exact identifiers already "
                    "present in the user query.\n"
                    "- Translate semantic intent into useful "
                    "English/code terminology when helpful.\n"
                    "- Do not answer the question.\n"
                    "- Do not invent repository-specific "
                    "function names, class names, file names, "
                    "or identifiers that were not present in "
                    "the original query.\n"
                    "- Each rewrite must express the same "
                    "information need from a different lexical "
                    "angle.\n"
                    f"- Return at most {self.max_rewrites} "
                    "rewrites.\n"
                    "- Return ONLY a JSON array of strings."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

    def _parse_rewrites(
        self,
        *,
        content: str,
        original_query: str,
    ) -> list[str]:
        payload = self._strip_code_fence(content.strip())

        try:
            parsed = json.loads(payload)

        except json.JSONDecodeError as error:
            raise QueryExpansionError(
                "LLM query expansion returned " "invalid JSON"
            ) from error

        if not isinstance(parsed, list):
            raise QueryExpansionError("LLM query expansion must " "return a JSON array")

        rewrites: list[str] = []
        seen = {original_query.casefold()}

        for item in parsed:
            if not isinstance(item, str):
                raise QueryExpansionError("Every query rewrite must " "be a string")

            rewrite = item.strip()

            if not rewrite:
                continue

            key = rewrite.casefold()

            if key in seen:
                continue

            seen.add(key)
            rewrites.append(rewrite)

            if len(rewrites) >= self.max_rewrites:
                break

        return rewrites

    @staticmethod
    def _strip_code_fence(
        text: str,
    ) -> str:
        if not text.startswith("```"):
            return text

        lines = text.splitlines()

        if len(lines) < 3:
            return text

        if not lines[-1].strip().startswith("```"):
            return text

        return "\n".join(lines[1:-1]).strip()
