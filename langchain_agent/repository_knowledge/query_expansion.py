from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from langchain_agent.repository_knowledge.errors import QueryExpansionError
from langchain_agent.repository_knowledge.ports import QueryExpander


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


class QueryExpansionPayload(BaseModel):
    """Structured supplemental queries returned by the chat model."""

    rewrites: list[str] = Field(
        default_factory=list,
        description=(
            "Supplemental code-repository search queries. "
            "Do not repeat the original query."
        ),
    )


class IdentityQueryExpander:
    """Baseline expander that leaves retrieval on the original query only."""

    def expand(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[str]:
        _validate_request(query, limit)
        return ()


class LLMQueryExpander:
    """Generate repository-search rewrites with an injected chat model."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
    ) -> None:
        self.model = model
        self._structured_model = model.with_structured_output(
            QueryExpansionPayload,
            include_raw=True,
        )

    def expand(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[str]:
        normalized_query = _validate_request(query, limit)

        messages = [
            (
                "system",
                self._system_prompt(limit),
            ),
            (
                "human",
                normalized_query,
            ),
        ]

        try:
            result = self._structured_model.invoke(
                messages,
                config={
                    "run_name": "rag_query_expansion",
                    "tags": [
                        "rag",
                        "query-expansion",
                    ],
                },
            )
        except (AssertionError, TypeError):
            raise
        except Exception as error:
            raise QueryExpansionError(
                "The query-expansion model request failed"
            ) from error

        rewrites = self._read_payload(result)
        return self._sanitize_rewrites(
            rewrites=rewrites,
            original_query=normalized_query,
            limit=limit,
        )

    @staticmethod
    def _system_prompt(limit: int) -> str:
        return (
            "You generate supplemental search queries for a Python source-code "
            "repository. Your goal is retrieval, not answering the question.\n\n"
            "Rules:\n"
            "- Preserve exact identifiers already present in the user query.\n"
            "- Translate semantic intent into useful English code terminology "
            "when helpful.\n"
            "- Express the same information need from different lexical angles.\n"
            "- Do not answer the question.\n"
            "- Do not invent repository-specific files, functions, classes, or "
            "identifiers absent from the original query.\n"
            "- Do not repeat the original query.\n"
            f"- Return at most {limit} supplemental queries."
        )

    @staticmethod
    def _read_payload(result: Any) -> Sequence[str]:
        if not isinstance(result, dict):
            raise QueryExpansionError(
                "The query-expansion model returned an unexpected envelope"
            )

        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            error = QueryExpansionError(
                "The query-expansion model returned invalid structured output"
            )
            if isinstance(parsing_error, BaseException):
                raise error from parsing_error
            raise error

        payload = result.get("parsed")

        if isinstance(payload, QueryExpansionPayload):
            return payload.rewrites

        if isinstance(payload, dict):
            rewrites = payload.get("rewrites")
            if isinstance(rewrites, list):
                return rewrites

        raise QueryExpansionError(
            "The query-expansion model returned no parsed rewrites"
        )

    @staticmethod
    def _sanitize_rewrites(
        *,
        rewrites: Sequence[str],
        original_query: str,
        limit: int,
    ) -> tuple[str, ...]:
        output: list[str] = []
        seen = {original_query.casefold()}

        for item in rewrites:
            if not isinstance(item, str):
                raise QueryExpansionError(
                    "Every supplemental query must be a string"
                )

            rewrite = item.strip()
            if not rewrite:
                continue

            key = rewrite.casefold()
            if key in seen:
                continue

            seen.add(key)
            output.append(rewrite)

            if len(output) >= limit:
                break

        return tuple(output)


class FallbackQueryExpander:
    """Fall back only for expected query-expansion failures."""

    def __init__(
        self,
        *,
        primary: QueryExpander,
        fallback: QueryExpander,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def expand(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[str]:
        try:
            return self.primary.expand(
                query,
                limit=limit,
            )
        except QueryExpansionError as error:
            logger.warning(
                "Query expansion failed; using the configured fallback: %s",
                error,
            )
            return self.fallback.expand(
                query,
                limit=limit,
            )


def _validate_request(query: str, limit: int) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be empty")

    return normalized_query
