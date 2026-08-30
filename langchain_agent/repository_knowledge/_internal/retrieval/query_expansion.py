from __future__ import annotations

from typing import Protocol


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
