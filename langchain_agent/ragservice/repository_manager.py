from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from langchain_agent.ragservice.embedding import (
    SentenceTransformerEmbeddingClient,
)
from langchain_agent.ragservice.interfaces import EmbeddingClient
from langchain_agent.ragservice.python_repository_rag import (
    PythonRepositoryRAG,
    RetrievalMode,
)

from langchain_agent.ragservice.repository_snapshot import (
    RepositoryChangedDuringIndexingError,
)

ToolResult = dict[str, Any]


class RepositoryKnowledgeManager:
    """
    管理 Agent 当前使用的代码仓库 RAG 索引。

    一个 Manager 当前维护一个活动仓库。

    index_repository_knowledge():
        建立或切换活动仓库索引。

    search_repository_knowledge():
        查询当前活动仓库。
    """

    def __init__(
        self,
        model_name: str = ("BAAI/bge-small-zh-v1.5"),
        max_chunk_characters: int = 2400,
        overlap_lines: int = 8,
        max_context_characters: int = 8000,
        max_context_items: int = 5,
        max_tool_result_characters: int = 10000,
        max_evidence_content_characters: int = 2400,
        show_progress_bar: bool = False,
        device: str | None = None,
        retrieval_mode: RetrievalMode = "fast",
    ) -> None:
        if max_tool_result_characters <= 0:
            raise ValueError("max_tool_result_characters " "must be greater than 0")

        if max_evidence_content_characters <= 0:
            raise ValueError(
                "max_evidence_content_characters " "must be greater than 0"
            )

        self.model_name = model_name
        self.max_chunk_characters = max_chunk_characters
        self.overlap_lines = overlap_lines
        self.max_context_characters = max_context_characters
        self.max_context_items = max_context_items
        self.show_progress_bar = show_progress_bar
        self.max_tool_result_characters = max_tool_result_characters
        self.max_evidence_content_characters = max_evidence_content_characters
        self.device = device
        self.retrieval_mode = retrieval_mode

        # 模型在第一次构建索引时才加载。
        self._embedding_client: EmbeddingClient | None = None

        self._repository_rag: PythonRepositoryRAG | None = None

        self._repository_path: Path | None = None

    @property
    def is_indexed(self) -> bool:
        return self._repository_rag is not None and self._repository_rag.is_indexed

    @property
    def repository_rag(self) -> PythonRepositoryRAG:
        if self._repository_rag is None:
            raise RuntimeError("Repository knowledge index is not ready.")

        return self._repository_rag

    def index_repository_knowledge(
        self,
        repository_path: str,
    ) -> ToolResult:
        """
        为本地 Python 仓库建立语义索引。

        构建成功后，该仓库成为当前活动知识库。
        """

        if not isinstance(repository_path, str):
            raise TypeError("repository_path must be a string")

        stripped_path = repository_path.strip()

        if not stripped_path:
            return self._failure(
                error_type="invalid_repository_path",
                message=("repository_path cannot be empty"),
            )

        resolved_path = Path(stripped_path).resolve()

        if not resolved_path.exists():
            return self._failure(
                error_type="repository_not_found",
                message=("Repository does not exist: " f"{resolved_path}"),
            )

        if not resolved_path.is_dir():
            return self._failure(
                error_type="not_a_directory",
                message=("Repository path is not a directory: " f"{resolved_path}"),
            )

        embedding_client = self._get_embedding_client()

        # 先构造一套候选 RAG。
        # 只有索引完整建立成功后，才替换当前活动仓库。
        candidate_rag = PythonRepositoryRAG(
            repository_path=resolved_path,
            embedding_client=embedding_client,
            max_chunk_characters=(self.max_chunk_characters),
            overlap_lines=self.overlap_lines,
            max_context_characters=(self.max_context_characters),
            max_context_items=(self.max_context_items),
            device=self.device,
            retrieval_mode=self.retrieval_mode,
        )

        index_directory = resolved_path / ".rag_index"

        try:
            ready_index = candidate_rag.ensure_index(index_directory)

        except RepositoryChangedDuringIndexingError as error:
            return self._failure(
                error_type="repository_changed_during_indexing",
                message=str(error),
            )

        index_result = ready_index.index
        index_source = ready_index.source
        rebuild_reason = ready_index.rebuild_reason

        if index_result.is_empty:
            return self._failure(
                error_type="empty_repository_index",
                message=(
                    "No indexable Python code was found "
                    f"in repository: {resolved_path}"
                ),
            )

        # 到这里说明候选索引已经完整建立。
        self._repository_rag = candidate_rag
        self._repository_path = resolved_path

        result: ToolResult = {
            "success": True,
            "repository_path": str(resolved_path),
            "index_directory": str(index_directory),
            "index_source": index_source,
            "document_count": (index_result.document_count),
            "chunk_count": (index_result.chunk_count),
            "vector_dimension": (index_result.vector_dimension),
            "message": ("Repository knowledge index " "is ready."),
        }

        if rebuild_reason is not None:
            result["rebuild_reason"] = rebuild_reason

        return result

    def search_repository_knowledge(
        self,
        query: str,
        top_k: int = 8,
    ) -> ToolResult:
        """
        在当前活动代码仓库中进行混合检索。

        内部结合：
        - 向量语义检索
        - BM25 关键词检索
        - RRF 排名融合

        只返回经过预算控制的代码证据，
        不调用额外 LLM。
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        stripped_query = query.strip()

        if not stripped_query:
            return self._failure(
                error_type="invalid_query",
                message="query cannot be empty",
            )

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if not 1 <= top_k <= 12:
            return self._failure(
                error_type="invalid_top_k",
                message=("top_k must be between 1 and 12"),
            )

        repository_rag = self._repository_rag

        if repository_rag is None:
            return self._failure(
                error_type="repository_not_indexed",
                message=(
                    "No repository knowledge index "
                    "is currently available. Call "
                    "index_repository_knowledge first."
                ),
            )

        search_response = repository_rag.search(
            query=stripped_query,
            top_k=top_k,
        )

        selected_items = list(search_response.sources)

        evidence: list[ToolResult] = []

        for item in selected_items:
            serialized_item = self._serialize_evidence(item)

            candidate_evidence = [
                *evidence,
                serialized_item,
            ]

            candidate_result = self._build_search_result(
                query=stripped_query,
                retrieved_count=len(search_response.search_results),
                selected_count=len(selected_items),
                context_character_count=(search_response.context.character_count),
                evidence=candidate_evidence,
            )

            if self._encoded_size(candidate_result) > self.max_tool_result_characters:
                break

            evidence.append(serialized_item)

        return self._build_search_result(
            query=stripped_query,
            retrieved_count=len(search_response.search_results),
            selected_count=len(selected_items),
            context_character_count=(search_response.context.character_count),
            evidence=evidence,
        )

    def _build_search_result(
        self,
        *,
        query: str,
        retrieved_count: int,
        selected_count: int,
        context_character_count: int,
        evidence: list[ToolResult],
    ) -> ToolResult:
        omitted_evidence_count = selected_count - len(evidence)

        content_was_truncated = any(
            item.get(
                "content_truncated",
                False,
            )
            for item in evidence
        )

        return {
            "success": True,
            "repository_path": str(self._repository_path),
            "query": query,
            "retrieved_count": retrieved_count,
            "selected_count": selected_count,
            "evidence_count": len(evidence),
            "omitted_evidence_count": (omitted_evidence_count),
            "context_character_count": (context_character_count),
            "truncated": (omitted_evidence_count > 0 or content_was_truncated),
            "evidence": evidence,
        }

    def _get_embedding_client(
        self,
    ) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = SentenceTransformerEmbeddingClient(
                model_name=self.model_name,
                show_progress_bar=(self.show_progress_bar),
                device=self.device,
            )

        return self._embedding_client

    @staticmethod
    def _truncate_content(
        content: str,
        max_characters: int,
    ) -> tuple[str, bool]:
        if len(content) <= max_characters:
            return content, False

        suffix = "\n...[content truncated; " "use read_file for full context]"

        available_characters = max(
            1,
            max_characters - len(suffix),
        )

        cut_position = content.rfind(
            "\n",
            0,
            available_characters,
        )

        # 如果前半部分没有合适的换行，
        # 说明可能存在一条特别长的代码行。
        if cut_position < available_characters // 2:
            cut_position = available_characters

        truncated_content = content[:cut_position].rstrip() + suffix

        return truncated_content, True

    @staticmethod
    def _encoded_size(
        result: ToolResult,
    ) -> int:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        return len(encoded)

    def _serialize_evidence(
        self,
        item,
    ) -> ToolResult:
        chunk = item.chunk
        metadata = chunk.metadata

        content, content_truncated = self._truncate_content(
            chunk.content,
            self.max_evidence_content_characters,
        )

        evidence: ToolResult = {
            "source_id": item.context_id,
            "rank": item.retrieval_rank,
            "source": chunk.source,
            "content": content,
        }

        symbol = metadata.get("symbol")

        if isinstance(symbol, str) and symbol:
            evidence["symbol"] = symbol

        symbol_type = metadata.get("symbol_type")

        if isinstance(symbol_type, str) and symbol_type:
            evidence["symbol_type"] = symbol_type

        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")

        if isinstance(start_line, int) and isinstance(end_line, int):
            evidence["start_line"] = start_line
            evidence["end_line"] = end_line

        part_index = metadata.get("part_index")
        part_count = metadata.get("part_count")

        if (
            isinstance(part_index, int)
            and isinstance(part_count, int)
            and part_count > 1
        ):
            evidence["part"] = f"{part_index + 1}/" f"{part_count}"

        if content_truncated:
            evidence["content_truncated"] = True

        return evidence

    @staticmethod
    def _failure(
        error_type: str,
        message: str,
    ) -> ToolResult:
        return {
            "success": False,
            "error_type": error_type,
            "error": message,
        }
