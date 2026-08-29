import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from langchain_agent.repository_knowledge import (
    Evidence,
    IndexNotReadyError,
    InvalidRepositoryError,
    RepositoryKnowledgeConfig,
    RepositoryKnowledgeService,
)
from langchain_agent.repository_knowledge.cache import repository_cache_key


class FakeEmbeddingClient:
    model_id = "fake-embedding-v1"
    dimension = 4

    def __init__(self) -> None:
        self.document_batches = 0
        self.query_count = 0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_batches += 1
        return np.vstack([self._embed(text) for text in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self.query_count += 1
        return self._embed(query)

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        normalized = text.casefold()
        vector = np.asarray(
            [
                normalized.count("alpha") + 0.1,
                normalized.count("beta") + 0.1,
                normalized.count("permission") + 0.1,
                1.0,
            ],
            dtype=np.float32,
        )
        return vector / np.linalg.norm(vector)


def write_python_repository(root: Path, file_name: str, symbol: str) -> None:
    (root / file_name).write_text(
        f"def {symbol}():\n"
        f"    \"\"\"Return the {symbol} value.\"\"\"\n"
        f"    return \"{symbol}\"\n",
        encoding="utf-8",
    )


class RepositoryKnowledgeConfigTests(unittest.TestCase):
    def test_rejects_invalid_retrieval_configuration(self) -> None:
        invalid_values = [
            {"retrieval_mode": "unknown"},
            {"max_chunk_characters": 0},
            {"overlap_lines": -1},
            {"max_context_characters": 0},
            {"max_context_items": 0},
        ]

        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RepositoryKnowledgeConfig(**values)


class RepositoryCacheKeyTests(unittest.TestCase):
    def test_key_is_stable_for_the_same_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()

            self.assertEqual(
                repository_cache_key(repository),
                repository_cache_key(repository.resolve()),
            )

    def test_same_repository_name_in_different_locations_does_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_repository = root / "first" / "repository"
            second_repository = root / "second" / "repository"
            first_repository.mkdir(parents=True)
            second_repository.mkdir(parents=True)

            self.assertNotEqual(
                repository_cache_key(first_repository),
                repository_cache_key(second_repository),
            )


class RepositoryKnowledgeServiceTests(unittest.TestCase):
    def test_search_requires_explicit_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            write_python_repository(repository, "alpha.py", "alpha")

            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=root / "agent-cache" / "alpha",
                embedding_client=FakeEmbeddingClient(),
            )

            self.assertFalse(service.is_ready)

            with self.assertRaises(IndexNotReadyError):
                service.search("alpha")

    def test_prepare_and_search_return_public_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            index_path = root / "agent-cache" / "alpha"
            repository.mkdir()
            write_python_repository(repository, "alpha.py", "alpha")

            service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=index_path,
                embedding_client=FakeEmbeddingClient(),
            )

            ready = service.prepare()

            self.assertTrue(service.is_ready)
            self.assertEqual(ready.source, "rebuilt")
            self.assertEqual(ready.repository_path, repository.resolve())
            self.assertEqual(ready.index_path, index_path.resolve())
            self.assertGreater(ready.chunk_count, 0)
            self.assertTrue((index_path / "manifest.json").is_file())
            self.assertFalse((repository / ".rag_index").exists())

            response = service.search("  alpha  ", top_k=3)

            self.assertEqual(response.query, "alpha")
            self.assertEqual(response.repository_path, repository.resolve())
            self.assertGreater(response.retrieved_count, 0)
            self.assertGreater(len(response.evidence), 0)
            self.assertTrue(response.context)
            self.assertEqual(
                response.context_character_count,
                len(response.context),
            )

            evidence = response.evidence[0]
            self.assertIsInstance(evidence, Evidence)
            self.assertEqual(evidence.source, "alpha.py")
            self.assertIn("def alpha", evidence.content)
            self.assertEqual(evidence.symbol, "alpha")
            self.assertEqual(evidence.symbol_type, "function")
            self.assertEqual(evidence.start_line, 1)

    def test_new_service_loads_an_existing_application_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            index_path = root / "agent-cache" / "alpha"
            repository.mkdir()
            write_python_repository(repository, "alpha.py", "alpha")
            embedding_client = FakeEmbeddingClient()

            first_service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=index_path,
                embedding_client=embedding_client,
            )
            first_service.prepare()
            document_batches_after_build = embedding_client.document_batches

            second_service = RepositoryKnowledgeService(
                repository_path=repository,
                index_path=index_path,
                embedding_client=embedding_client,
            )
            ready = second_service.prepare()

            self.assertEqual(ready.source, "loaded")
            self.assertEqual(
                embedding_client.document_batches,
                document_batches_after_build,
            )

    def test_services_share_embedding_client_and_isolate_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpha_repository = root / "alpha-repository"
            beta_repository = root / "beta-repository"
            alpha_repository.mkdir()
            beta_repository.mkdir()
            write_python_repository(alpha_repository, "alpha.py", "alpha")
            write_python_repository(beta_repository, "beta.py", "beta")
            embedding_client = FakeEmbeddingClient()

            alpha_service = RepositoryKnowledgeService(
                repository_path=alpha_repository,
                index_path=root / "agent-cache" / "alpha",
                embedding_client=embedding_client,
            )
            beta_service = RepositoryKnowledgeService(
                repository_path=beta_repository,
                index_path=root / "agent-cache" / "beta",
                embedding_client=embedding_client,
            )

            self.assertIs(alpha_service.embedding_client, embedding_client)
            self.assertIs(beta_service.embedding_client, embedding_client)

            alpha_service.prepare()
            beta_service.prepare()

            alpha_result = alpha_service.search("alpha")
            beta_result = beta_service.search("beta")

            self.assertEqual(alpha_result.evidence[0].source, "alpha.py")
            self.assertEqual(beta_result.evidence[0].source, "beta.py")
            self.assertNotEqual(
                alpha_result.repository_path,
                beta_result.repository_path,
            )

    def test_rejects_a_missing_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaises(InvalidRepositoryError):
                RepositoryKnowledgeService(
                    repository_path=root / "missing",
                    index_path=root / "agent-cache",
                    embedding_client=FakeEmbeddingClient(),
                )


if __name__ == "__main__":
    unittest.main()
