from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_agent.ragservice.interfaces import EmbeddingClient

from langchain_agent.ragservice.code_prompt_builder import (
    CodeRAGPromptBuilder,
)
from langchain_agent.ragservice.context_builder import ContextBuilder
from langchain_agent.ragservice.embedding import (
    SentenceTransformerEmbeddingClient,
)
from langchain_agent.ragservice.generator import RAGGenerator
from langchain_agent.ragservice.indexer import RAGIndexer
from langchain_agent.ragservice.models import (
    IndexBuildResult,
    IndexReadyResult,
    RAGAnswer,
    RAGSearchResponse,
)
from langchain_agent.ragservice.python_chunker import (
    PythonASTChunker,
)
from langchain_agent.ragservice.python_loader import (
    PythonDocumentLoader,
)
from langchain_agent.ragservice.retriever import VectorRetriever
from langchain_agent.ragservice.service import NaiveRAG
from langchain_agent.ragservice.vector_store import (
    InMemoryVectorStore,
)

from langchain_agent.ragservice.index_storage import (
    IndexCompatibilityError,
    IndexCorruptionError,
    RAGIndexStorage,
)

from langchain_agent.ragservice.repository_snapshot import (
    RepositoryChangedDuringIndexingError,
    RepositorySnapshot,
    RepositorySnapshotBuilder,
    describe_repository_changes,
)

from langchain_agent.ragservice.bm25 import (
    BM25Retriever,
    InMemoryBM25Index,
)
from langchain_agent.ragservice.code_tokenizer import (
    CodeTokenizer,
)
from langchain_agent.ragservice.hybrid_retriever import (
    HybridRetriever,
)

from langchain_agent.ragservice.query_expansion import (
    LLMQueryExpander,
)

from langchain_agent.ragservice.multi_query_retriever import (
    MultiQueryRetriever,
)

from langchain_agent.ragservice.reranker import (
    CrossEncoderReranker,
    RerankingRetriever,
)

RetrievalMode = Literal[
    "fast",
    "quality",
]


class PythonRepositoryRAG:
    """
    面向本地 Python 仓库的完整 RAG。

    rebuild():
        扫描并建立代码索引。

    answer():
        对已经建立的代码索引进行问答。
    """

    def __init__(
        self,
        repository_path: str | Path,
        model_name: str = ("BAAI/bge-small-zh-v1.5"),
        embedding_client: EmbeddingClient | None = None,
        max_chunk_characters: int = 2400,
        overlap_lines: int = 8,
        max_context_characters: int = 10000,
        max_context_items: int = 6,
        show_progress_bar: bool = False,
        device: str | None = None,
        retrieval_mode: RetrievalMode = "fast",
    ) -> None:
        self.repository_path = Path(repository_path).resolve()

        if not self.repository_path.exists():
            raise FileNotFoundError(
                "Repository does not exist: " f"{self.repository_path}"
            )

        if not self.repository_path.is_dir():
            raise NotADirectoryError(
                "Repository path is not a directory: " f"{self.repository_path}"
            )

        if retrieval_mode not in (
            "fast",
            "quality",
        ):
            raise ValueError("retrieval_mode must be " "'fast' or 'quality'")

        self.retrieval_mode = retrieval_mode

        self.loader = PythonDocumentLoader()

        self.snapshot_builder = RepositorySnapshotBuilder(loader=self.loader)

        self.max_chunk_characters = max_chunk_characters
        self.overlap_lines = overlap_lines

        self.chunker = PythonASTChunker(
            max_chunk_characters=self.max_chunk_characters,
            overlap_lines=self.overlap_lines,
        )

        if embedding_client is None:
            self.embedding_client = SentenceTransformerEmbeddingClient(
                model_name=model_name,
                show_progress_bar=(show_progress_bar),
                device=device,
            )
            self.embedding_model_id = model_name

        else:
            self.embedding_client = embedding_client

            self.embedding_model_id = str(
                getattr(
                    embedding_client,
                    "model_name",
                    type(embedding_client).__qualname__,
                )
            )

        self.vector_store = InMemoryVectorStore(
            dimension=(self.embedding_client.dimension)
        )

        self.code_tokenizer = CodeTokenizer()

        self.bm25_index = InMemoryBM25Index(
            tokenizer=self.code_tokenizer,
            k1=1.5,
            b=0.75,
        )

        self.indexer = RAGIndexer(
            loader=self.loader,
            chunker=self.chunker,
            embedding_client=(self.embedding_client),
            vector_store=self.vector_store,
        )

        self.vector_retriever = VectorRetriever(
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )

        self.bm25_retriever = BM25Retriever(index=self.bm25_index)

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=(self.vector_retriever),
            lexical_retriever=(self.bm25_retriever),
            rrf_k=60,
            candidate_multiplier=3,
        )

        self.query_expander = LLMQueryExpander(max_rewrites=2)

        self.multi_query_retriever = MultiQueryRetriever(
            base_retriever=(self.hybrid_retriever),
            query_expander=(self.query_expander),
            rrf_k=60,
        )

        self.reranker: CrossEncoderReranker | None = None
        self.rerank_candidate_count = 30

        if self.retrieval_mode == "fast":
            self.retriever = self.multi_query_retriever

        else:
            reranker = CrossEncoderReranker(
                # model_name=("BAAI/bge-reranker-v2-m3"),
                model_name=r"C:\Users\31972\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3",
                batch_size=8,
                max_length=512,
                device=device,
                show_progress_bar=False,
            )

            self.retriever = RerankingRetriever(
                base_retriever=(self.multi_query_retriever),
                reranker=reranker,
                candidate_count=self.rerank_candidate_count,
            )

        self.context_builder = ContextBuilder(
            max_context_characters=(max_context_characters),
            max_items=max_context_items,
            include_scores=False,
        )

        self.generator = RAGGenerator(prompt_builder=(CodeRAGPromptBuilder()))

        self.service = NaiveRAG(
            retriever=self.retriever,
            context_builder=(self.context_builder),
            generator=self.generator,
        )

    @property
    def is_indexed(self) -> bool:
        return not self.vector_store.is_empty and not self.bm25_index.is_empty

    def ensure_index(
        self,
        index_directory: str | Path,
        max_attempts: int = 2,
    ) -> IndexReadyResult:
        storage = RAGIndexStorage(index_directory)

        rebuild_reason: str | None = None

        if storage.exists():
            try:
                index_result = self._load_index(index_directory)

                return IndexReadyResult(
                    index=index_result,
                    source="disk",
                )

            except (
                IndexCorruptionError,
                IndexCompatibilityError,
            ) as error:
                rebuild_reason = str(error)

        index_result = self._rebuild_and_save(
            index_directory=index_directory,
            max_attempts=max_attempts,
        )

        return IndexReadyResult(
            index=index_result,
            source="rebuilt",
            rebuild_reason=rebuild_reason,
        )

    def rebuild(self) -> IndexBuildResult:
        index_result = self.indexer.rebuild_directory(self.repository_path)

        self.bm25_index.replace(self.vector_store.chunks)

        return index_result

    def answer(
        self,
        question: str,
        top_k: int = 12,
        minimum_score: float | None = None,
    ) -> RAGAnswer:
        if not self.is_indexed:
            raise RuntimeError(
                "Repository index has not been built. "
                "Call rebuild() before answer()."
            )

        return self.service.answer(
            question=question,
            top_k=top_k,
            minimum_score=minimum_score,
        )

    def search(
        self,
        query: str,
        top_k: int = 12,
        minimum_score: float | None = None,
    ) -> RAGSearchResponse:
        if not self.is_indexed:
            raise RuntimeError(
                "Repository index has not been built. "
                "Call rebuild() before search()."
            )

        return self.service.search(
            query=query,
            top_k=top_k,
            minimum_score=minimum_score,
        )

    def _rebuild_and_save(
        self,
        index_directory: str | Path,
        max_attempts: int = 2,
    ) -> IndexBuildResult:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0")

        snapshot_before = self.build_repository_snapshot()

        for _ in range(max_attempts):
            index_result = self.rebuild()

            snapshot_after = self.build_repository_snapshot()

            if snapshot_before == snapshot_after:
                if not index_result.is_empty:
                    self._save_index(
                        index_directory=index_directory,
                        repository_files=snapshot_after,
                    )

                return index_result

            snapshot_before = snapshot_after

        raise RepositoryChangedDuringIndexingError(
            "Repository Python files kept " "changing while the index was built."
        )

    def _save_index(
        self,
        index_directory: str | Path,
        repository_files: RepositorySnapshot,
    ) -> dict[str, object]:
        """
        将当前内存索引保存到磁盘。
        """
        chunks, vectors = self.vector_store.snapshot()

        document_count = len({chunk.document_id for chunk in chunks})

        storage = RAGIndexStorage(index_directory)

        return storage.save(
            repository_path=(self.repository_path),
            repository_files=(repository_files),
            embedding_model=(self.embedding_model_id),
            vector_dimension=(self.embedding_client.dimension),
            chunker_type=(type(self.chunker).__name__),
            chunker_config={
                "max_chunk_characters": (self.max_chunk_characters),
                "overlap_lines": (self.overlap_lines),
            },
            document_count=document_count,
            chunks=chunks,
            vectors=vectors,
        )

    def _load_index(
        self,
        index_directory: str | Path,
    ) -> IndexBuildResult:
        storage = RAGIndexStorage(index_directory)

        # 这里返回的索引已经是：
        # 格式合法、文件完整、内部自洽。
        loaded_index = storage.load()

        current_repository_files = self.build_repository_snapshot()

        # 这里只判断：
        # 它能否用于当前仓库和当前配置。
        self._validate_index_compatibility(
            manifest=loaded_index.manifest,
            current_repository_files=(current_repository_files),
        )

        self.vector_store.replace(
            chunks=loaded_index.chunks,
            vectors=loaded_index.vectors,
        )

        self.bm25_index.replace(loaded_index.chunks)

        return IndexBuildResult(
            source=str(self.repository_path),
            document_count=(loaded_index.manifest["document_count"]),
            chunk_count=len(loaded_index.chunks),
            vector_dimension=(self.embedding_client.dimension),
        )

    def build_repository_snapshot(
        self,
    ) -> RepositorySnapshot:
        return self.snapshot_builder.build(self.repository_path)

    def _validate_index_compatibility(
        self,
        manifest: dict[str, object],
        current_repository_files: RepositorySnapshot,
    ) -> None:
        stored_repository_path = Path(str(manifest["repository_path"])).resolve()

        if stored_repository_path != self.repository_path:
            raise IndexCompatibilityError(
                "Index belongs to a different "
                "repository: "
                f"{stored_repository_path}"
            )

        stored_model = manifest["embedding_model"]

        if stored_model != self.embedding_model_id:
            raise IndexCompatibilityError(
                "Index embedding model does "
                "not match the current model: "
                f"{stored_model} != "
                f"{self.embedding_model_id}"
            )

        stored_dimension = manifest["vector_dimension"]

        if stored_dimension != self.embedding_client.dimension:
            raise IndexCompatibilityError(
                "Index vector dimension does "
                "not match the current model: "
                f"{stored_dimension} != "
                f"{self.embedding_client.dimension}"
            )

        expected_chunker_type = type(self.chunker).__name__

        if manifest["chunker_type"] != expected_chunker_type:
            raise IndexCompatibilityError(
                "Index chunker type does "
                "not match current chunker: "
                f"{manifest['chunker_type']} != "
                f"{expected_chunker_type}"
            )

        expected_chunker_config = {
            "max_chunk_characters": (self.max_chunk_characters),
            "overlap_lines": (self.overlap_lines),
        }

        if manifest["chunker_config"] != expected_chunker_config:
            raise IndexCompatibilityError(
                "Index chunker configuration " "does not match current " "configuration"
            )

        stored_repository_files = manifest["repository_files"]

        if stored_repository_files != current_repository_files:
            change_description = describe_repository_changes(
                stored_snapshot=(stored_repository_files),
                current_snapshot=(current_repository_files),
            )

            raise IndexCompatibilityError(
                "Repository Python sources "
                "changed after indexing: "
                f"{change_description}"
            )
