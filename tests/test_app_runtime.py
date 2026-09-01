import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_agent.app.bootstrap import bootstrap_application
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
                paths=AppPaths.under(root / ".agent"),
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
    def test_application_paths_follow_data_config_and_cache_lifetimes(self):
        paths = AppPaths(
            data_dir=Path("data"),
            config_dir=Path("config"),
            cache_dir=Path("cache"),
        )

        self.assertEqual(paths.checkpoint_path, Path("data/checkpoints.sqlite"))
        self.assertEqual(paths.session_path, Path("data/sessions.json"))
        self.assertEqual(paths.mcp_config_path, Path("config/mcp.json"))
        self.assertEqual(paths.environment_path, Path("config/.env"))
        self.assertEqual(paths.index_root, Path("cache/indexes"))

    def test_user_defaults_use_platform_directories(self):
        platform_paths = SimpleNamespace(
            user_data_path=Path("C:/app-data"),
            user_config_path=Path("C:/app-config"),
            user_cache_path=Path("C:/app-cache"),
        )

        with patch(
            "langchain_agent.app.config.PlatformDirs",
            return_value=platform_paths,
        ) as platform_dirs:
            paths = AppPaths.user_default()

        platform_dirs.assert_called_once_with(
            "langchain-agent",
            appauthor=False,
        )
        self.assertEqual(paths.data_dir, Path("C:/app-data"))
        self.assertEqual(paths.config_dir, Path("C:/app-config"))
        self.assertEqual(paths.cache_dir, Path("C:/app-cache"))

    def test_under_co_locates_paths_for_injected_runtimes(self):
        paths = AppPaths.under(Path("runtime-data"))

        self.assertEqual(paths.data_dir, Path("runtime-data"))
        self.assertEqual(paths.config_dir, Path("runtime-data"))
        self.assertEqual(paths.cache_dir, Path("runtime-data"))


class ApplicationBootstrapTests(unittest.IsolatedAsyncioTestCase):
    @patch("langchain_agent.app.bootstrap.create_model")
    @patch("langchain_agent.app.bootstrap.load_dotenv")
    async def test_loads_explicit_environment_before_creating_models(
        self,
        load_dotenv,
        create_model,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(paths=AppPaths.under(directory))
            create_model.side_effect = RuntimeError("model creation reached")

            with self.assertRaisesRegex(RuntimeError, "model creation reached"):
                async with bootstrap_application(config):
                    self.fail("bootstrap should stop at model creation")

        load_dotenv.assert_called_once_with(
            dotenv_path=config.paths.environment_path,
            override=False,
        )
        create_model.assert_called_once_with(thinking=False)


if __name__ == "__main__":
    unittest.main()
