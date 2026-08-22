from dataclasses import dataclass, field
from typing import Any, Literal

Metadata = dict[str, Any]


@dataclass
class Document:
    """
    一份完整的原始文档。
    第一版中，一份Markdown 或 TXT文件对应一个Document。
    """

    id: str
    content: str
    source: str
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Chunk:
    """
    从Document中切分出的可检索知识单元。
    一个Documnet通常对应多个Chunk。
    """

    id: str
    document_id: str
    content: str
    source: str
    index: int
    start_char: int  # 起始字符位置
    end_char: int  # 结束字符位置，右开
    metadata: Metadata = field(default_factory=dict)
    embedding_content: str | None = None

    @property
    def content_for_embedding(self) -> str:
        return (
            self.embedding_content
            if self.embedding_content is not None
            else self.content
        )


@dataclass
class SearchResult:
    """
    一次检索返回的结果。

    score 的含义由 Retriever 决定：

    VectorRetriever:
        余弦相似度。

    BM25Retriever:
        BM25 相关性分数。

    HybridRetriever:
        RRF 融合分数。
    """

    chunk: Chunk
    score: float
    rank: int


@dataclass
class ContextItem:
    """
    最终放入 RAG Context 的一条资料。

    context_id 是面向LLM的资料编号，
    例如 source_1、source_2。
    """

    context_id: str
    chunk: Chunk
    score: float
    retrieval_rank: int


@dataclass
class BuiltContext:
    """
    ContextBuilder 的输出。

    text：
        实际交给 LLM 的资料文本。

    items：
        保留结构化资料，方便后面生成来源列表、检查引用和调试。

    character_count：
        当前 Context的字符数量。
    """

    text: str
    items: list[ContextItem]
    character_count: int

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass
class RAGAnswer:
    """
    一次完整 RAG 问答的结果。

    answer：
        LLM 最终生成的回答。

    context：
        实际交给 LLM 的检索上下文。

    search_results：
        Retriever 最初返回的结果，方便调试检索排名。
    """

    question: str
    answer: str
    context: BuiltContext
    search_results: list[SearchResult]

    @property
    def sources(self) -> list[ContextItem]:
        return self.context.items


@dataclass
class RAGSearchResponse:
    """
    一次完整的 RAG 检索结果，但不调用生成模型。

    query:
        实际检索的问题。

    context:
        经过筛选、编号和预算控制后的上下文。

    search_results:
        Retriever 返回的 Top-k 检索结果
    """

    query: str
    context: BuiltContext
    search_results: list[SearchResult]

    @property
    def sources(self) -> list[ContextItem]:
        return self.context.items


@dataclass
class IndexBuildResult:
    """
    一次索引构建的统计结果。
    """

    source: str
    document_count: int
    chunk_count: int
    vector_dimension: int

    @property
    def is_empty(self) -> bool:
        return self.chunk_count == 0


@dataclass(frozen=True)
class IndexReadyResult:
    """
    展示index是由disk加载还是重建。
    """

    index: IndexBuildResult
    source: Literal["disk", "rebuilt"]
    rebuild_reason: str | None = None
