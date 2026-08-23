from __future__ import annotations

from langchain_agent.ragservice.context_builder import ContextBuilder
from langchain_agent.ragservice.generator import RAGGenerator
from langchain_agent.ragservice.models import (
    RAGAnswer,
    RAGSearchResponse,
)
from langchain_agent.ragservice.interfaces import Retriever


class NaiveRAG:
    """
    最基础的完整 RAG 流程。

    search():
        只执行检索和 Context 构造。

    answer():
        在 search() 的基础上调用 LLM 生成答案。
    """

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: RAGGenerator,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def search(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> RAGSearchResponse:
        search_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            minimum_score=minimum_score,
        )

        context = self.context_builder.build(search_results)

        return RAGSearchResponse(
            query=query,
            context=context,
            search_results=search_results,
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> RAGAnswer:
        search_response = self.search(
            query=question,
            top_k=top_k,
            minimum_score=minimum_score,
        )

        context = search_response.context

        if context.is_empty:
            return RAGAnswer(
                question=question,
                answer=("知识库中没有检索到可用于回答" "该问题的资料。"),
                context=context,
                search_results=(search_response.search_results),
            )

        generated_answer = self.generator.generate(
            question=question,
            context=context,
        )

        return RAGAnswer(
            question=question,
            answer=generated_answer,
            context=context,
            search_results=(search_response.search_results),
        )
