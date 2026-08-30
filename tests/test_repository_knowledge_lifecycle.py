import tempfile
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from langchain_agent.repository_knowledge import (
    IndexBuildError,
    IndexNotReadyError,
    RepositoryChangedDuringIndexingError,
    RepositoryKnowledgeService,
    RepositoryKnowledgeStatus,
)


class DummyEmbeddingClient:
    model_id = "dummy-embedding-v1"
    dimension = 4

    def embed_documents(self, texts):
        raise AssertionError("embedding should not be called")

    def embed_query(self, query):
        raise AssertionError("embedding should not be called")


class MutatingFailingBackend:
    is_indexed = True

    def ensure_index(self, index_path):
        raise RepositoryChangedDuringIndexingError(
            "repository changed after the in-memory index was populated"
        )


class CountingEmbeddingClient:
    model_id = "counting-embedding-v1"
    dimension = 2

    def __init__(self) -> None:
        self.document_batches = 0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_batches += 1
        return np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def write_repository_file(repository: Path, value: str) -> None:
    (repository / "example.py").write_text(
        "def repository_value():\n" f"    return {value!r}\n",
        encoding="utf-8",
    )


def prepared_index(snapshot=None):
    return SimpleNamespace(
        index=SimpleNamespace(
            is_empty=False,
            document_count=1,
            chunk_count=1,
            vector_dimension=4,
        ),
        repository_snapshot={} if snapshot is None else snapshot,
        source="rebuilt",
        rebuild_reason=None,
    )


class RepositoryKnowledgeLifecycleTests(unittest.TestCase):
    def test_failed_candidate_backend_is_never_published_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=DummyEmbeddingClient(),
            )

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                return_value=MutatingFailingBackend(),
            ):
                with self.assertRaises(RepositoryChangedDuringIndexingError):
                    service.prepare()

            self.assertFalse(service.is_ready)
            self.assertEqual(
                service.status,
                RepositoryKnowledgeStatus.FAILED,
            )
            self.assertIsInstance(
                service.last_error,
                RepositoryChangedDuringIndexingError,
            )

            with self.assertRaises(IndexNotReadyError):
                service.search("repository")

    def test_failed_prepare_can_be_retried_with_a_new_candidate(self) -> None:
        class SuccessfulBackend:
            is_indexed = True

            def ensure_index(self, index_path):
                return prepared_index()

            def build_repository_snapshot(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=DummyEmbeddingClient(),
            )

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                side_effect=[
                    MutatingFailingBackend(),
                    SuccessfulBackend(),
                ],
            ):
                with self.assertRaises(RepositoryChangedDuringIndexingError):
                    service.prepare()

                result = service.prepare()

            self.assertTrue(service.is_ready)
            self.assertEqual(service.status, RepositoryKnowledgeStatus.READY)
            self.assertEqual(result.chunk_count, 1)
            self.assertIsNone(service.last_error)

    def test_prepare_reuses_fresh_index_and_rebuilds_after_source_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            write_repository_file(repository, "first")
            embedding_client = CountingEmbeddingClient()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=embedding_client,
            )

            first_result = service.prepare()
            batches_after_first_prepare = embedding_client.document_batches
            repeated_result = service.prepare()

            self.assertIs(repeated_result, first_result)
            self.assertEqual(
                embedding_client.document_batches,
                batches_after_first_prepare,
            )

            write_repository_file(repository, "second")
            refreshed_result = service.prepare()

            self.assertTrue(service.is_ready)
            self.assertEqual(refreshed_result.source, "rebuilt")
            self.assertGreater(
                embedding_client.document_batches,
                batches_after_first_prepare,
            )
            self.assertIn(
                "second",
                service.search("repository value").context,
            )

    def test_failed_refresh_disables_the_previous_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            write_repository_file(repository, "first")
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=CountingEmbeddingClient(),
            )
            service.prepare()
            write_repository_file(repository, "second")

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                return_value=MutatingFailingBackend(),
            ):
                with self.assertRaises(RepositoryChangedDuringIndexingError):
                    service.prepare()

            self.assertEqual(service.status, RepositoryKnowledgeStatus.FAILED)
            self.assertFalse(service.is_ready)

            with self.assertRaises(IndexNotReadyError):
                service.search("repository value")

    def test_concurrent_prepare_builds_only_one_candidate(self) -> None:
        entered_prepare = Event()
        release_prepare = Event()

        class BlockingBackend:
            is_indexed = True

            def __init__(self) -> None:
                self.ensure_count = 0

            def ensure_index(self, index_path):
                self.ensure_count += 1
                entered_prepare.set()
                if not release_prepare.wait(timeout=5):
                    raise TimeoutError("test did not release prepare")
                return prepared_index()

            def build_repository_snapshot(self):
                return {}

        backend = BlockingBackend()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=DummyEmbeddingClient(),
            )

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                return_value=backend,
            ) as backend_factory:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(service.prepare)
                    self.assertTrue(entered_prepare.wait(timeout=5))
                    self.assertEqual(
                        service.status,
                        RepositoryKnowledgeStatus.PREPARING,
                    )
                    second_started = Event()

                    def prepare_second_time():
                        second_started.set()
                        return service.prepare()

                    second = executor.submit(prepare_second_time)
                    self.assertTrue(second_started.wait(timeout=5))
                    release_prepare.set()
                    first_result = first.result(timeout=5)
                    second_result = second.result(timeout=5)

            self.assertIs(second_result, first_result)
            self.assertEqual(backend.ensure_count, 1)
            self.assertEqual(backend_factory.call_count, 1)

    def test_os_error_becomes_index_build_error_with_original_cause(self) -> None:
        expected_error = PermissionError("cache directory is read-only")

        class PermissionFailingBackend:
            def ensure_index(self, index_path):
                raise expected_error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=DummyEmbeddingClient(),
            )

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                return_value=PermissionFailingBackend(),
            ):
                with self.assertRaises(IndexBuildError) as raised:
                    service.prepare()

            self.assertIs(raised.exception.__cause__, expected_error)
            self.assertIs(service.last_error, raised.exception)
            self.assertEqual(service.status, RepositoryKnowledgeStatus.FAILED)

    def test_unexpected_bug_is_not_translated_to_a_domain_error(self) -> None:
        expected_error = AssertionError("broken backend invariant")

        class BuggyBackend:
            def ensure_index(self, index_path):
                raise expected_error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "index",
                embedding_client=DummyEmbeddingClient(),
            )

            with patch(
                "langchain_agent.repository_knowledge._internal.backend."
                "PythonRepositoryKnowledgeBackend",
                return_value=BuggyBackend(),
            ):
                with self.assertRaises(AssertionError) as raised:
                    service.prepare()

            self.assertIs(raised.exception, expected_error)
            self.assertIs(service.last_error, expected_error)
            self.assertEqual(service.status, RepositoryKnowledgeStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
