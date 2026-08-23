from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from langchain_agent.ragservice.models import Chunk

FloatMatrix = NDArray[np.float32]

INDEX_SCHEMA_VERSION = 1


class IndexStorageError(RuntimeError):
    """
    索引持久化相关错误的基类。
    """


class IndexCorruptionError(IndexStorageError):
    """
    索引文件缺失、损坏或内部数据不一致。
    """


class IndexCompatibilityError(IndexStorageError):
    """
    索引存在，但配置与当前 RAG 不兼容。
    """


@dataclass(frozen=True)
class LoadedIndex:
    manifest: dict[str, Any]
    chunks: list[Chunk]
    vectors: FloatMatrix


class RAGIndexStorage:
    """
    将 RAG 索引保存到本地目录。

    文件结构：

        manifest.json
        chunks.json
        vectors.npy
    """

    MANIFEST_FILE = "manifest.json"
    CHUNKS_FILE = "chunks.json"
    VECTORS_FILE = "vectors.npy"

    def __init__(
        self,
        index_directory: str | Path,
    ) -> None:
        self.index_directory = Path(index_directory).resolve()

        self.manifest_path = self.index_directory / self.MANIFEST_FILE

        self.chunks_path = self.index_directory / self.CHUNKS_FILE

        self.vectors_path = self.index_directory / self.VECTORS_FILE

    def exists(self) -> bool:
        """
        三个文件全部存在时，才认为索引完整存在。
        """

        return (
            self.manifest_path.is_file()
            and self.chunks_path.is_file()
            and self.vectors_path.is_file()
        )

    def save(
        self,
        *,
        repository_path: str | Path,
        repository_files: dict[
            str,
            dict[str, Any],
        ],
        embedding_model: str,
        vector_dimension: int,
        chunker_type: str,
        chunker_config: dict[str, Any],
        document_count: int,
        chunks: list[Chunk],
        vectors: FloatMatrix,
    ) -> dict[str, Any]:
        """
        保存完整索引。

        manifest 最后替换，相当于索引写入的完成标志。
        """

        matrix = self._validate_index_data(
            chunks=chunks,
            vectors=vectors,
            expected_dimension=(vector_dimension),
        )

        chunk_payload = [self._serialize_chunk(chunk) for chunk in chunks]

        self.index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_manifest = self._temporary_path(self.manifest_path)

        temporary_chunks = self._temporary_path(self.chunks_path)

        temporary_vectors = self._temporary_path(self.vectors_path)

        try:
            self._write_json(
                temporary_chunks,
                chunk_payload,
            )

            self._write_vectors(
                temporary_vectors,
                matrix,
            )

            chunks_sha256 = self._calculate_sha256(temporary_chunks)

            vectors_sha256 = self._calculate_sha256(temporary_vectors)

            manifest: dict[str, Any] = {
                "schema_version": (INDEX_SCHEMA_VERSION),
                "created_at_utc": (datetime.now(timezone.utc).isoformat()),
                "repository_path": str(Path(repository_path).resolve()),
                "repository_file_count": len(repository_files),
                "repository_files": (repository_files),
                "embedding_model": embedding_model,
                "vector_dimension": (vector_dimension),
                "chunker_type": chunker_type,
                "chunker_config": chunker_config,
                "document_count": document_count,
                "chunk_count": len(chunks),
                "chunks_sha256": chunks_sha256,
                "vectors_sha256": vectors_sha256,
            }

            self._write_json(
                temporary_manifest,
                manifest,
            )

            # 先替换数据文件，最后替换 manifest。
            #
            # 如果程序在中间崩溃，
            # manifest 中的哈希与数据不一致，
            # 加载时会拒绝该索引。
            os.replace(
                temporary_chunks,
                self.chunks_path,
            )

            os.replace(
                temporary_vectors,
                self.vectors_path,
            )

            os.replace(
                temporary_manifest,
                self.manifest_path,
            )

            return manifest

        finally:
            self._remove_if_exists(temporary_manifest)
            self._remove_if_exists(temporary_chunks)
            self._remove_if_exists(temporary_vectors)

    def load(self) -> LoadedIndex:
        if not self.exists():
            raise FileNotFoundError(
                "RAG index is incomplete or "
                f"does not exist: "
                f"{self.index_directory}"
            )

        raw_manifest = self._read_json_object(self.manifest_path)

        manifest = self._validate_manifest_structure(raw_manifest)

        self._verify_file_hash(
            file_path=self.chunks_path,
            expected_hash=manifest["chunks_sha256"],
            file_name=self.CHUNKS_FILE,
        )

        self._verify_file_hash(
            file_path=self.vectors_path,
            expected_hash=manifest["vectors_sha256"],
            file_name=self.VECTORS_FILE,
        )

        chunk_payload = self._read_json_list(self.chunks_path)

        chunks = [self._deserialize_chunk(item) for item in chunk_payload]

        vectors = self._read_vectors(self.vectors_path)

        matrix = self._validate_index_data(
            chunks=chunks,
            vectors=vectors,
            expected_dimension=manifest["vector_dimension"],
        )

        if manifest["chunk_count"] != len(chunks):
            raise IndexCorruptionError(
                "Manifest chunk count does not "
                "match chunks.json: "
                f"{manifest['chunk_count']} != "
                f"{len(chunks)}"
            )

        document_ids = {chunk.document_id for chunk in chunks}

        if manifest["document_count"] != len(document_ids):
            raise IndexCorruptionError(
                "Manifest document count does "
                "not match loaded chunks: "
                f"{manifest['document_count']} != "
                f"{len(document_ids)}"
            )

        return LoadedIndex(
            manifest=manifest,
            chunks=chunks,
            vectors=matrix,
        )

    @staticmethod
    def _serialize_chunk(
        chunk: Chunk,
    ) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "document_id": (chunk.document_id),
            "content": chunk.content,
            "source": chunk.source,
            "index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata,
            "embedding_content": (chunk.embedding_content),
        }

    @staticmethod
    def _deserialize_chunk(
        payload: Any,
    ) -> Chunk:
        if not isinstance(payload, dict):
            raise IndexCorruptionError("Each chunks.json item must " "be an object")

        try:
            metadata = payload["metadata"]

            if not isinstance(
                metadata,
                dict,
            ):
                raise TypeError("metadata must be an object")

            embedding_content = payload.get("embedding_content")

            if embedding_content is not None and not isinstance(
                embedding_content,
                str,
            ):
                raise TypeError("embedding_content must " "be a string or null")

            return Chunk(
                id=str(payload["id"]),
                document_id=str(payload["document_id"]),
                content=str(payload["content"]),
                source=str(payload["source"]),
                index=int(payload["index"]),
                start_char=int(payload["start_char"]),
                end_char=int(payload["end_char"]),
                metadata=metadata,
                embedding_content=(embedding_content),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise IndexCorruptionError("Invalid Chunk data in " "chunks.json") from exc

    @staticmethod
    def _validate_index_data(
        *,
        chunks: list[Chunk],
        vectors: FloatMatrix,
        expected_dimension: int,
    ) -> FloatMatrix:
        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise IndexCorruptionError(
                "Index vectors must be a " "two-dimensional matrix"
            )

        if matrix.shape[0] != len(chunks):
            raise IndexCorruptionError(
                "Chunk and vector counts do "
                "not match: "
                f"{len(chunks)} chunks, "
                f"{matrix.shape[0]} vectors"
            )

        if matrix.shape[1] != expected_dimension:
            raise IndexCorruptionError(
                "Unexpected vector dimension: "
                f"expected "
                f"{expected_dimension}, "
                f"got {matrix.shape[1]}"
            )

        if not np.all(np.isfinite(matrix)):
            raise IndexCorruptionError("Index vectors contain " "NaN or infinity")

        return matrix

    def _validate_manifest_structure(
        self,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        schema_version = self._require_int(
            manifest,
            "schema_version",
            minimum=1,
        )

        if schema_version != INDEX_SCHEMA_VERSION:
            raise IndexCompatibilityError(
                "Unsupported RAG index schema: "
                f"expected {INDEX_SCHEMA_VERSION}, "
                f"got {schema_version}"
            )

        repository_path = self._require_string(
            manifest,
            "repository_path",
        )

        embedding_model = self._require_string(
            manifest,
            "embedding_model",
        )

        vector_dimension = self._require_int(
            manifest,
            "vector_dimension",
            minimum=1,
        )

        chunker_type = self._require_string(
            manifest,
            "chunker_type",
        )

        chunker_config = manifest.get("chunker_config")

        if not isinstance(
            chunker_config,
            dict,
        ):
            raise IndexCorruptionError("manifest.chunker_config " "must be an object")

        document_count = self._require_int(
            manifest,
            "document_count",
            minimum=0,
        )

        chunk_count = self._require_int(
            manifest,
            "chunk_count",
            minimum=0,
        )

        repository_file_count = self._require_int(
            manifest,
            "repository_file_count",
            minimum=0,
        )

        repository_files = self._validate_repository_files(
            manifest.get("repository_files")
        )

        if repository_file_count != len(repository_files):
            raise IndexCorruptionError(
                "manifest.repository_file_count " "does not match repository_files"
            )

        chunks_sha256 = self._require_string(
            manifest,
            "chunks_sha256",
        )

        vectors_sha256 = self._require_string(
            manifest,
            "vectors_sha256",
        )

        return {
            **manifest,
            "schema_version": schema_version,
            "repository_path": repository_path,
            "embedding_model": embedding_model,
            "vector_dimension": vector_dimension,
            "chunker_type": chunker_type,
            "chunker_config": chunker_config,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "repository_file_count": (repository_file_count),
            "repository_files": (repository_files),
            "chunks_sha256": chunks_sha256,
            "vectors_sha256": vectors_sha256,
        }

    @staticmethod
    def _validate_repository_files(
        payload: Any,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise IndexCorruptionError("manifest.repository_files " "must be an object")

        validated: dict[
            str,
            dict[str, Any],
        ] = {}

        for source, file_state in payload.items():
            if not isinstance(source, str):
                raise IndexCorruptionError("Repository file path " "must be a string")

            if not source:
                raise IndexCorruptionError("Repository file path " "cannot be empty")

            if not isinstance(
                file_state,
                dict,
            ):
                raise IndexCorruptionError("Repository file state " "must be an object")

            size = file_state.get("size")
            sha256 = file_state.get("sha256")

            if not isinstance(size, int) or size < 0:
                raise IndexCorruptionError(
                    "Repository file size " "must be a non-negative integer"
                )

            if not isinstance(sha256, str) or len(sha256) != 64:
                raise IndexCorruptionError(
                    "Repository file SHA-256 " "must be a 64-character string"
                )

            validated[source] = {
                "size": size,
                "sha256": sha256,
            }

        return validated

    @staticmethod
    def _require_string(
        payload: dict[str, Any],
        key: str,
    ) -> str:
        value = payload.get(key)

        if not isinstance(value, str):
            raise IndexCorruptionError(f"manifest.{key} " "must be a string")

        if not value:
            raise IndexCorruptionError(f"manifest.{key} " "cannot be empty")

        return value

    @staticmethod
    def _require_int(
        payload: dict[str, Any],
        key: str,
        minimum: int,
    ) -> int:
        value = payload.get(key)

        # bool 是 int 的子类，所以需要明确排除。
        if isinstance(value, bool) or not isinstance(value, int):
            raise IndexCorruptionError(f"manifest.{key} " "must be an integer")

        if value < minimum:
            raise IndexCorruptionError(f"manifest.{key} " f"must be at least {minimum}")

        return value

    @staticmethod
    def _write_json(
        path: Path,
        payload: Any,
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _write_vectors(
        path: Path,
        vectors: FloatMatrix,
    ) -> None:
        with path.open("wb") as file:
            np.save(
                file,
                vectors,
                allow_pickle=False,
            )

            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _read_json_object(
        path: Path,
    ) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise IndexCorruptionError(f"Cannot read index file: " f"{path}") from exc

        if not isinstance(payload, dict):
            raise IndexCorruptionError(f"Index file must contain " f"an object: {path}")

        return payload

    @staticmethod
    def _read_json_list(
        path: Path,
    ) -> list[Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise IndexCorruptionError(f"Cannot read index file: " f"{path}") from exc

        if not isinstance(payload, list):
            raise IndexCorruptionError(f"Index file must contain " f"a list: {path}")

        return payload

    @staticmethod
    def _read_vectors(
        path: Path,
    ) -> FloatMatrix:
        try:
            with path.open("rb") as file:
                vectors = np.load(
                    file,
                    allow_pickle=False,
                )

        except (
            OSError,
            ValueError,
            EOFError,
        ) as exc:
            raise IndexCorruptionError(f"Cannot read vectors: {path}") from exc

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    @staticmethod
    def _calculate_sha256(
        path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)

        return digest.hexdigest()

    def _verify_file_hash(
        self,
        *,
        file_path: Path,
        expected_hash: Any,
        file_name: str,
    ) -> None:
        if not isinstance(
            expected_hash,
            str,
        ):
            raise IndexCorruptionError(f"Missing hash for {file_name}")

        actual_hash = self._calculate_sha256(file_path)

        if actual_hash != expected_hash:
            raise IndexCorruptionError(f"Index file hash mismatch: " f"{file_name}")

    @staticmethod
    def _temporary_path(
        target_path: Path,
    ) -> Path:
        return target_path.with_name(f".{target_path.name}." f"{uuid4().hex}.tmp")

    @staticmethod
    def _remove_if_exists(
        path: Path,
    ) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
