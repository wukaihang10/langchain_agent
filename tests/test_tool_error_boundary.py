import unittest
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent.repository_knowledge import (
    RepositoryChangedDuringIndexingError,
    RepositoryKnowledgeError,
)
from langchain_agent.tool_errors import ToolErrorMiddleware
from langchain_agent.tools.errors import RepositoryToolError
from langchain_agent.tools.rag import search_repository_knowledge


@tool
def failing_repository_tool() -> str:
    """Exercise the expected repository failure boundary."""

    raise RepositoryToolError("repository is unavailable")


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class ToolInputSchemaTests(unittest.TestCase):
    def test_query_must_not_be_blank(self) -> None:
        with self.assertRaises(ValueError):
            search_repository_knowledge.tool_call_schema.model_validate(
                {
                    "query": "   ",
                    "top_k": 5,
                }
            )

    def test_top_k_must_stay_within_supported_range(self) -> None:
        for invalid_top_k in (0, 13):
            with self.subTest(top_k=invalid_top_k), self.assertRaises(ValueError):
                search_repository_knowledge.tool_call_schema.model_validate(
                    {
                        "query": "permission middleware",
                        "top_k": invalid_top_k,
                    }
                )


class RepositoryKnowledgeToolTests(unittest.TestCase):
    def test_tool_prepares_service_before_first_search(self) -> None:
        class FakeRepositoryKnowledge:
            is_ready = False

            def __init__(self) -> None:
                self.prepare_count = 0
                self.search_calls = []

            def prepare(self) -> None:
                self.prepare_count += 1
                self.is_ready = True

            def search(self, query: str, *, top_k: int):
                self.search_calls.append((query, top_k))
                return SimpleNamespace(context="repository evidence")

        repository_knowledge = FakeRepositoryKnowledge()
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                repository_knowledge=repository_knowledge,
            )
        )

        result = search_repository_knowledge.func(
            query="permission middleware",
            runtime=runtime,
            top_k=7,
        )

        self.assertEqual(result, "repository evidence")
        self.assertEqual(repository_knowledge.prepare_count, 1)
        self.assertEqual(
            repository_knowledge.search_calls,
            [("permission middleware", 7)],
        )

    def test_expected_prepare_error_bubbles_out_of_tool(self) -> None:
        expected_error = RepositoryChangedDuringIndexingError(
            "repository kept changing"
        )

        class FailingRepositoryKnowledge:
            is_ready = False

            def prepare(self) -> None:
                raise expected_error

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                repository_knowledge=FailingRepositoryKnowledge(),
            )
        )

        with self.assertRaises(RepositoryChangedDuringIndexingError) as raised:
            search_repository_knowledge.func(
                query="permission middleware",
                runtime=runtime,
                top_k=5,
            )

        self.assertIs(raised.exception, expected_error)


class ToolErrorMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.middleware = ToolErrorMiddleware()
        self.request = SimpleNamespace(
            tool_call={
                "id": "call-1",
                "name": "search_repository_knowledge",
                "args": {},
            }
        )

    async def test_expected_domain_error_becomes_tool_message(self) -> None:
        async def failing_handler(request):
            raise RepositoryKnowledgeError("repository is unavailable")

        result = await self.middleware.awrap_tool_call(
            self.request,
            failing_handler,
        )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertIn("repository is unavailable", result.content)

    async def test_expected_repository_tool_error_becomes_tool_message(self) -> None:
        async def failing_handler(request):
            raise RepositoryToolError("file does not exist")

        result = await self.middleware.awrap_tool_call(
            self.request,
            failing_handler,
        )

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertIn("file does not exist", result.content)

    async def test_unexpected_programming_error_is_not_hidden(self) -> None:
        async def broken_handler(request):
            raise KeyError("context")

        with self.assertRaises(KeyError):
            await self.middleware.awrap_tool_call(
                self.request,
                broken_handler,
            )


class AgentToolErrorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_continues_after_expected_tool_failure(self) -> None:
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "failing_repository_tool",
                            "args": {},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="I can continue after the tool failure."),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[failing_repository_tool],
            middleware=[ToolErrorMiddleware()],
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Search the repository."),
                ]
            }
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].status, "error")
        self.assertEqual(
            result["messages"][-1].content,
            "I can continue after the tool failure.",
        )


if __name__ == "__main__":
    unittest.main()
