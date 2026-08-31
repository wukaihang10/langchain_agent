import tempfile
import unittest
from pathlib import Path

from langchain_agent.app.config import AppConfig, AppPaths
from langchain_agent.app.repository_knowledge import RepositoryKnowledgeProvider
from langchain_agent.app.session_runtime import (
    build_session_runtime,
    delete_session,
)
from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.persistence.sessions import SessionStore
from langchain_agent.repository_knowledge import RepositoryKnowledgeConfig


class DummyEmbeddingClient:
    def embed_documents(self, texts):
        raise AssertionError("embedding should stay lazy in this test")

    def embed_query(self, text):
        raise AssertionError("embedding should stay lazy in this test")


class DummyQueryExpander:
    def expand(self, query, *, limit):
        return [query]


class RepositoryKnowledgeProviderTests(unittest.TestCase):
    def test_services_are_cached_per_repository_and_share_embedding_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository_a = root / "a" / "repository"
            repository_b = root / "b" / "repository"
            repository_a.mkdir(parents=True)
            repository_b.mkdir(parents=True)
            factory_calls = 0

            def create_embedding_client():
                nonlocal factory_calls
                factory_calls += 1
                return DummyEmbeddingClient()

            provider = RepositoryKnowledgeProvider(
                index_root=root / "indexes",
                embedding_client_factory=create_embedding_client,
                query_expander=DummyQueryExpander(),
                config=RepositoryKnowledgeConfig(),
            )

            service_a = provider.get(repository_a)
            same_service_a = provider.get(repository_a / ".")
            service_b = provider.get(repository_b)

            self.assertIs(service_a, same_service_a)
            self.assertIsNot(service_a, service_b)
            self.assertEqual(factory_calls, 1)


class SessionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_preserves_thread_metadata_and_permission_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            store = SessionStore(root / "sessions.json")
            session = store.create(
                name="demo",
                repository_path=str(repository),
            )
            knowledge_service = object()

            class Provider:
                def get(self, repository_path):
                    self.repository_path = repository_path
                    return knowledge_service

            provider = Provider()
            config = AppConfig(
                permission_mode=PermissionMode.READ_ONLY,
                paths=AppPaths(root / ".agent"),
            )

            runtime = build_session_runtime(
                session=session,
                repository_path=repository,
                session_store=store,
                repository_knowledge_provider=provider,
                config=config,
            )

            self.assertEqual(runtime.session.thread_id, session.thread_id)
            self.assertEqual(runtime.context.repository_path, str(repository.resolve()))
            self.assertIs(runtime.context.repository_knowledge, knowledge_service)
            self.assertEqual(
                runtime.context.permission_mode,
                PermissionMode.READ_ONLY,
            )
            self.assertEqual(
                runtime.invoke_config["configurable"]["thread_id"],
                session.thread_id,
            )
            self.assertEqual(
                runtime.invoke_config["metadata"]["permission_mode"],
                "read_only",
            )
            self.assertEqual(provider.repository_path, repository.resolve())

    async def test_delete_removes_checkpoint_before_session_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions.json")
            session = store.create(
                name="demo",
                repository_path=str(root),
            )
            calls: list[tuple[str, str]] = []

            class Checkpointer:
                async def adelete_thread(self, thread_id):
                    calls.append(("checkpoint", thread_id))
                    self.metadata_still_exists = store.get(thread_id) is not None

            checkpointer = Checkpointer()
            deleted = await delete_session(
                session=session,
                session_store=store,
                checkpointer=checkpointer,
            )

            self.assertTrue(deleted)
            self.assertTrue(checkpointer.metadata_still_exists)
            self.assertEqual(calls, [("checkpoint", session.thread_id)])
            self.assertIsNone(store.get(session.thread_id))


class AppConfigTests(unittest.TestCase):
    def test_all_application_paths_derive_from_agent_directory(self):
        paths = AppPaths(Path("runtime-data"))

        self.assertEqual(paths.checkpoint_path, Path("runtime-data/checkpoints.sqlite"))
        self.assertEqual(paths.session_path, Path("runtime-data/sessions.json"))
        self.assertEqual(paths.mcp_config_path, Path("runtime-data/mcp.json"))
        self.assertEqual(paths.index_root, Path("runtime-data/indexes"))


if __name__ == "__main__":
    unittest.main()
