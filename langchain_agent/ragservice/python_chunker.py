from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from langgraph_agent.ragservice.chunker import TextChunker
from langgraph_agent.ragservice.models import Chunk, Document


@dataclass(frozen=True)
class PythonSpan:
    """
    一个 Python 语义符号所占据的源码范围。

    行号采用 1-based，并且 end_line 包含在范围内。
    """

    start_line: int
    end_line: int
    symbol: str
    symbol_type: str
    parent_symbol: str | None = None

    # 函数或类定义头的结束行。
    # 用于让后续子片段的 Embedding 仍能看到函数签名。
    header_end_line: int | None = None


@dataclass(frozen=True)
class PythonPart:
    """
    一个 PythonSpan 进一步切分后的代码片段。
    """

    start_line: int
    end_line: int
    part_index: int
    part_count: int


class PythonASTChunker:
    """
    使用 AST 按类、函数和方法切分 Python 源码。

    当一个符号超过 max_chunk_characters 时，
    会继续按完整代码行切分，并保留若干重叠行。
    """

    def __init__(
        self,
        max_chunk_characters: int = 2400,
        overlap_lines: int = 8,
        minimum_part_ratio: float = 0.6,
        fallback_chunker: TextChunker | None = None,
    ) -> None:
        if max_chunk_characters <= 0:
            raise ValueError("max_chunk_characters must be positive")

        if overlap_lines < 0:
            raise ValueError("overlap_lines cannot be negative")

        if not 0 < minimum_part_ratio <= 1:
            raise ValueError("minimum_part_ratio must be within (0, 1]")

        self.max_chunk_characters = max_chunk_characters
        self.overlap_lines = overlap_lines
        self.minimum_part_ratio = minimum_part_ratio

        self.fallback_chunker = (
            fallback_chunker
            if fallback_chunker is not None
            else TextChunker(
                chunk_size=1200,
                chunk_overlap=150,
            )
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def split_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        if not document.content.strip():
            return []

        try:
            syntax_tree = ast.parse(
                document.content,
                filename=document.source,
            )
        except SyntaxError:
            return self._fallback_split(document)

        line_offsets = self._build_line_offsets(document.content)

        spans = self._collect_module_spans(syntax_tree)

        spans.sort(
            key=lambda span: (
                span.start_line,
                span.end_line,
                span.symbol,
            )
        )

        chunks: list[Chunk] = []
        chunk_index = 0

        for span in spans:
            parts = self._split_span(
                document=document,
                span=span,
                line_offsets=line_offsets,
            )

            for part in parts:
                chunk = self._create_chunk(
                    document=document,
                    span=span,
                    part=part,
                    index=chunk_index,
                    line_offsets=line_offsets,
                )

                if not chunk.content.strip():
                    continue

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _collect_module_spans(
        self,
        module: ast.Module,
    ) -> list[PythonSpan]:
        spans: list[PythonSpan] = []
        ordinary_nodes: list[ast.stmt] = []

        def flush_ordinary_nodes() -> None:
            if not ordinary_nodes:
                return

            spans.append(
                PythonSpan(
                    start_line=self._node_start_line(ordinary_nodes[0]),
                    end_line=self._node_end_line(ordinary_nodes[-1]),
                    symbol="<module>",
                    symbol_type="module",
                )
            )

            ordinary_nodes.clear()

        for node in module.body:
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                flush_ordinary_nodes()

                spans.append(
                    self._create_function_span(
                        node=node,
                        symbol=node.name,
                        parent_symbol=None,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                flush_ordinary_nodes()

                spans.extend(
                    self._collect_class_spans(
                        node=node,
                        parent_symbol=None,
                    )
                )

            else:
                ordinary_nodes.append(node)

        flush_ordinary_nodes()

        return spans

    def _collect_class_spans(
        self,
        node: ast.ClassDef,
        parent_symbol: str | None,
    ) -> list[PythonSpan]:
        spans: list[PythonSpan] = []

        class_symbol = f"{parent_symbol}.{node.name}" if parent_symbol else node.name

        class_start = self._node_start_line(node)

        if node.body:
            first_body_line = self._node_start_line(node.body[0])

            class_header_end = max(
                class_start,
                first_body_line - 1,
            )
        else:
            class_header_end = self._node_end_line(node)

        spans.append(
            PythonSpan(
                start_line=class_start,
                end_line=class_header_end,
                symbol=class_symbol,
                symbol_type="class",
                parent_symbol=parent_symbol,
                header_end_line=class_header_end,
            )
        )

        ordinary_nodes: list[ast.stmt] = []

        def flush_class_body() -> None:
            if not ordinary_nodes:
                return

            spans.append(
                PythonSpan(
                    start_line=self._node_start_line(ordinary_nodes[0]),
                    end_line=self._node_end_line(ordinary_nodes[-1]),
                    symbol=(f"{class_symbol}.<class_body>"),
                    symbol_type="class_body",
                    parent_symbol=class_symbol,
                )
            )

            ordinary_nodes.clear()

        for child in node.body:
            if isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                flush_class_body()

                method_symbol = f"{class_symbol}.{child.name}"

                spans.append(
                    self._create_function_span(
                        node=child,
                        symbol=method_symbol,
                        parent_symbol=class_symbol,
                    )
                )

            elif isinstance(child, ast.ClassDef):
                flush_class_body()

                spans.extend(
                    self._collect_class_spans(
                        node=child,
                        parent_symbol=class_symbol,
                    )
                )

            else:
                ordinary_nodes.append(child)

        flush_class_body()

        return spans

    def _create_function_span(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        symbol: str,
        parent_symbol: str | None,
    ) -> PythonSpan:
        start_line = self._node_start_line(node)
        end_line = self._node_end_line(node)

        if node.body:
            first_body_line = self._node_start_line(node.body[0])

            header_end_line = max(
                start_line,
                first_body_line - 1,
            )
        else:
            header_end_line = start_line

        if isinstance(
            node,
            ast.AsyncFunctionDef,
        ):
            symbol_type = "async_method" if parent_symbol else "async_function"
        else:
            symbol_type = "method" if parent_symbol else "function"

        return PythonSpan(
            start_line=start_line,
            end_line=end_line,
            symbol=symbol,
            symbol_type=symbol_type,
            parent_symbol=parent_symbol,
            header_end_line=header_end_line,
        )

    def _split_span(
        self,
        document: Document,
        span: PythonSpan,
        line_offsets: list[int],
    ) -> list[PythonPart]:
        start_char = line_offsets[span.start_line - 1]

        end_char = line_offsets[span.end_line]

        if end_char - start_char <= self.max_chunk_characters:
            return [
                PythonPart(
                    start_line=span.start_line,
                    end_line=span.end_line,
                    part_index=0,
                    part_count=1,
                )
            ]

        ranges: list[tuple[int, int]] = []
        current_start = span.start_line

        while current_start <= span.end_line:
            hard_end = self._find_hard_end_line(
                start_line=current_start,
                maximum_end_line=span.end_line,
                line_offsets=line_offsets,
            )

            preferred_end = self._find_preferred_end_line(
                text=document.content,
                start_line=current_start,
                hard_end_line=hard_end,
                line_offsets=line_offsets,
            )

            ranges.append(
                (
                    current_start,
                    preferred_end,
                )
            )

            if preferred_end >= span.end_line:
                break

            next_start = preferred_end - self.overlap_lines + 1

            # 必须保证至少前进一行，
            # 避免 overlap 过大导致死循环。
            current_start = max(
                current_start + 1,
                next_start,
            )

        part_count = len(ranges)

        return [
            PythonPart(
                start_line=start_line,
                end_line=end_line,
                part_index=index,
                part_count=part_count,
            )
            for index, (
                start_line,
                end_line,
            ) in enumerate(ranges)
        ]

    def _find_hard_end_line(
        self,
        start_line: int,
        maximum_end_line: int,
        line_offsets: list[int],
    ) -> int:
        start_char = line_offsets[start_line - 1]

        end_line = start_line

        while end_line <= maximum_end_line:
            candidate_end_char = line_offsets[end_line]

            candidate_size = candidate_end_char - start_char

            if candidate_size > self.max_chunk_characters:
                break

            end_line += 1

        # 如果单独一行已经超过最大长度，
        # 仍然完整保留该行，不从行中间切断。
        return max(
            start_line,
            end_line - 1,
        )

    def _find_preferred_end_line(
        self,
        text: str,
        start_line: int,
        hard_end_line: int,
        line_offsets: list[int],
    ) -> int:
        """
        在允许范围的后半部分寻找空行，
        尽量在代码段落边界结束。
        """

        line_count = hard_end_line - start_line + 1

        minimum_lines = max(
            1,
            int(line_count * self.minimum_part_ratio),
        )

        minimum_end_line = start_line + minimum_lines - 1

        for line_number in range(
            hard_end_line,
            minimum_end_line - 1,
            -1,
        ):
            line_start = line_offsets[line_number - 1]

            line_end = line_offsets[line_number]

            line = text[line_start:line_end]

            if not line.strip():
                return line_number

        return hard_end_line

    def _create_chunk(
        self,
        document: Document,
        span: PythonSpan,
        part: PythonPart,
        index: int,
        line_offsets: list[int],
    ) -> Chunk:
        start_char = line_offsets[part.start_line - 1]

        end_char = line_offsets[part.end_line]

        content = document.content[start_char:end_char].rstrip()

        adjusted_end_char = start_char + len(content)

        definition_header = self._extract_definition_header(
            document=document,
            span=span,
            line_offsets=line_offsets,
        )

        embedding_content = self._build_embedding_content(
            document=document,
            span=span,
            part=part,
            definition_header=(definition_header),
            content=content,
        )

        chunk_id = self._build_chunk_id(
            document_id=document.id,
            span=span,
            part=part,
            content=content,
        )

        metadata = {
            **document.metadata,
            "language": "python",
            "symbol": span.symbol,
            "symbol_type": span.symbol_type,
            "parent_symbol": span.parent_symbol,
            "start_line": part.start_line,
            "end_line": part.end_line,
            "symbol_start_line": span.start_line,
            "symbol_end_line": span.end_line,
            "part_index": part.part_index,
            "part_count": part.part_count,
            "is_partial": (part.part_count > 1),
            "chunk_index": index,
            "start_char": start_char,
            "end_char": adjusted_end_char,
            "character_count": len(content),
            "ast_parsed": True,
        }

        return Chunk(
            id=chunk_id,
            document_id=document.id,
            content=content,
            source=document.source,
            index=index,
            start_char=start_char,
            end_char=adjusted_end_char,
            metadata=metadata,
            embedding_content=embedding_content,
        )

    @staticmethod
    def _extract_definition_header(
        document: Document,
        span: PythonSpan,
        line_offsets: list[int],
    ) -> str | None:
        if span.header_end_line is None:
            return None

        if span.header_end_line < span.start_line:
            return None

        start_char = line_offsets[span.start_line - 1]

        end_char = line_offsets[span.header_end_line]

        header = document.content[start_char:end_char].rstrip()

        return header or None

    @staticmethod
    def _build_embedding_content(
        document: Document,
        span: PythonSpan,
        part: PythonPart,
        definition_header: str | None,
        content: str,
    ) -> str:
        lines = [
            "语言：Python",
            f"文件：{document.source}",
            f"符号：{span.symbol}",
            f"类型：{span.symbol_type}",
            ("完整符号范围：" f"{span.start_line}-" f"{span.end_line}"),
            ("当前片段：" f"{part.part_index + 1}/" f"{part.part_count}"),
            ("当前片段行号：" f"{part.start_line}-" f"{part.end_line}"),
        ]

        if definition_header:
            lines.extend(
                [
                    "",
                    "定义头：",
                    definition_header,
                ]
            )

        lines.extend(
            [
                "",
                "代码片段：",
                content,
            ]
        )

        return "\n".join(lines)

    def _fallback_split(
        self,
        document: Document,
    ) -> list[Chunk]:
        chunks = self.fallback_chunker.split_document(document)

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "language": "python",
                    "symbol": "<unknown>",
                    "symbol_type": ("text_fallback"),
                    "parent_symbol": None,
                    "part_index": 0,
                    "part_count": 1,
                    "is_partial": False,
                    "ast_parsed": False,
                }
            )

            chunk.embedding_content = (
                "语言：Python\n"
                f"文件：{chunk.source}\n"
                "类型：AST 解析失败后的文本片段\n\n"
                f"{chunk.content}"
            )

        return chunks

    @staticmethod
    def _build_line_offsets(
        text: str,
    ) -> list[int]:
        offsets = [0]

        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))

        # 文件最后没有换行时，累计长度仍然正确。
        if offsets[-1] < len(text):
            offsets.append(len(text))

        return offsets

    @staticmethod
    def _node_start_line(
        node: ast.AST,
    ) -> int:
        lineno = getattr(
            node,
            "lineno",
            None,
        )

        if lineno is None:
            raise ValueError("AST node has no start line")

        decorators = getattr(
            node,
            "decorator_list",
            [],
        )

        decorator_lines = [
            decorator.lineno
            for decorator in decorators
            if hasattr(
                decorator,
                "lineno",
            )
        ]

        if decorator_lines:
            return min(
                lineno,
                *decorator_lines,
            )

        return lineno

    @staticmethod
    def _node_end_line(
        node: ast.AST,
    ) -> int:
        end_lineno = getattr(
            node,
            "end_lineno",
            None,
        )

        if end_lineno is None:
            raise ValueError("AST node has no end line")

        return end_lineno

    @staticmethod
    def _build_chunk_id(
        document_id: str,
        span: PythonSpan,
        part: PythonPart,
        content: str,
    ) -> str:
        raw_id = (
            f"{document_id}\0"
            f"{span.symbol}\0"
            f"{span.symbol_type}\0"
            f"{part.start_line}\0"
            f"{part.end_line}\0"
            f"{content}"
        )

        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

        return (
            f"{document_id}:" f"{span.symbol}:" f"part_{part.part_index}:" f"{digest}"
        )
