import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from evals.environments.repository_fact import (
    RepositoryFactModels,
    open_repository_fact_environment,
)
from langchain_agent.app.config import AppConfig, AppPaths
from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.integrations.git import collect_file_editions
from langchain_agent.repository_knowledge import IdentityQueryExpander


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class DummyEmbeddingClient:
    def embed_documents(self, texts):
        raise AssertionError("embedding should stay lazy in this test")

    def embed_query(self, text):
        raise AssertionError("embedding should stay lazy in this test")


def build_models() -> RepositoryFactModels:
    models = [
        ToolCallingFakeModel(
            responses=[AIMessage(content="fixture answer")],
        )
        for _ in range(4)
    ]
    return RepositoryFactModels(
        primary=models[0],
        summary=models[1],
        researcher=models[2],
        reviewer=models[3],
    )


class RepositoryFactEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_clean_isolated_production_agent_environment(self):
        configured_paths = AppPaths(
            data_dir=Path("user-data"),
            config_dir=Path("user-config"),
            cache_dir=Path("user-cache"),
        )
        repository_path = None
        runtime_root = None

        async with open_repository_fact_environment(
            models=build_models(),
            embedding_client_factory=DummyEmbeddingClient,
            query_expander=IdentityQueryExpander(),
            config=AppConfig(
                permission_mode=PermissionMode.DEFAULT,
                paths=configured_paths,
                agent_version="environment-test-version",
            ),
        ) as environment:
            repository_path = environment.repository_path
            runtime_root = environment.runtime_root

            self.assertTrue((repository_path / ".git").is_dir())
            self.assertTrue(
                (
                    repository_path / "src" / "harbor_tasks" / "authorization.py"
                ).is_file()
            )
            self.assertEqual(
                collect_file_editions(str(repository_path))["edited_file_list"],
                [],
            )
            self.assertEqual(
                environment.config.permission_mode,
                PermissionMode.READ_ONLY,
            )
            self.assertEqual(environment.config.paths, AppPaths.under(runtime_root))
            self.assertNotEqual(environment.config.paths, configured_paths)

            output = await environment.target(
                {
                    "question": "Is an unknown tool allowed?",
                }
            )

            self.assertEqual(output["answer"], "fixture answer")
            self.assertEqual(output["git_audit_status"], "available")
            self.assertEqual(output["edited_files"], [])

        self.assertIsNotNone(repository_path)
        self.assertIsNotNone(runtime_root)
        self.assertFalse(repository_path.exists())
        self.assertFalse(runtime_root.exists())


if __name__ == "__main__":
    unittest.main()
