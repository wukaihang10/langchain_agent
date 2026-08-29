import unittest
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from langchain_agent.permissions.registry import ToolPolicyRegistry
from langchain_agent.permissions.types import (
    ToolCategory,
    ToolPolicy,
    ToolRisk,
)
from langchain_agent.tool_retry import (
    build_mcp_no_retry_failure_middleware,
    build_mcp_retry_middleware,
    is_transient_mcp_error,
)
from langchain_agent.tools.errors import RepositoryToolError


def build_policy(*, idempotent: bool, side_effect: bool) -> ToolPolicy:
    return ToolPolicy(
        category=(ToolCategory.WRITE if side_effect else ToolCategory.READ),
        idempotent=idempotent,
        side_effect=side_effect,
        risk=ToolRisk.LOW,
    )


def build_request(tool_name: str):
    return SimpleNamespace(
        tool=None,
        tool_call={
            "id": "call-1",
            "name": tool_name,
            "args": {},
        },
    )


class MCPRetryClassificationTests(unittest.TestCase):
    def test_only_transport_errors_are_retryable(self) -> None:
        self.assertTrue(is_transient_mcp_error(TimeoutError("timeout")))
        self.assertTrue(is_transient_mcp_error(ConnectionError("disconnected")))
        self.assertFalse(is_transient_mcp_error(ValueError("invalid input")))
        self.assertFalse(is_transient_mcp_error(RepositoryToolError("missing file")))

    def test_exception_group_requires_every_error_to_be_transient(self) -> None:
        transient_group = ExceptionGroup(
            "transport failures",
            [TimeoutError("timeout"), ConnectionError("disconnected")],
        )
        mixed_group = ExceptionGroup(
            "mixed failures",
            [TimeoutError("timeout"), ValueError("invalid response")],
        )

        self.assertTrue(is_transient_mcp_error(transient_group))
        self.assertFalse(is_transient_mcp_error(mixed_group))


class MCPRetryMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.safe_tool = SimpleNamespace(name="demo_search")
        self.side_effect_tool = SimpleNamespace(name="demo_create")
        self.non_idempotent_tool = SimpleNamespace(name="demo_random")
        self.unknown_tool = SimpleNamespace(name="demo_unknown")

        registry = ToolPolicyRegistry(
            {
                "demo_search": build_policy(
                    idempotent=True,
                    side_effect=False,
                ),
                "demo_create": build_policy(
                    idempotent=True,
                    side_effect=True,
                ),
                "demo_random": build_policy(
                    idempotent=False,
                    side_effect=False,
                ),
            }
        )

        self.middleware = build_mcp_retry_middleware(
            mcp_tools=[
                self.safe_tool,
                self.side_effect_tool,
                self.non_idempotent_tool,
                self.unknown_tool,
            ],
            policy_registry=registry,
            max_retries=2,
            initial_delay=0,
            jitter=False,
        )
        self.no_retry_middleware = build_mcp_no_retry_failure_middleware(
            mcp_tools=[
                self.safe_tool,
                self.side_effect_tool,
                self.non_idempotent_tool,
                self.unknown_tool,
            ],
            policy_registry=registry,
        )

        self.assertIsNotNone(self.middleware)
        self.assertIsNotNone(self.no_retry_middleware)

    async def test_safe_tool_retries_transient_failure(self) -> None:
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise TimeoutError("temporary timeout")

            return ToolMessage(
                content="success",
                tool_call_id="call-1",
            )

        result = await self.middleware.awrap_tool_call(
            build_request("demo_search"),
            handler,
        )

        self.assertEqual(call_count, 3)
        self.assertEqual(result.content, "success")

    async def test_exhausted_retries_become_tool_error(self) -> None:
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("service unavailable")

        result = await self.middleware.awrap_tool_call(
            build_request("demo_search"),
            handler,
        )

        self.assertEqual(call_count, 3)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertIn("automatic retries", result.content)

    async def test_unsafe_or_unknown_tools_are_reported_without_retry(self) -> None:
        for tool_name in (
            "demo_create",
            "demo_random",
            "demo_unknown",
        ):
            call_count = 0

            async def handler(request):
                nonlocal call_count
                call_count += 1
                raise TimeoutError("temporary timeout")

            with self.subTest(tool=tool_name):
                result = await self.no_retry_middleware.awrap_tool_call(
                    build_request(tool_name),
                    handler,
                )

            self.assertEqual(call_count, 1)
            self.assertIsInstance(result, ToolMessage)
            self.assertEqual(result.status, "error")
            self.assertIn("not automatically retried", result.content)
            self.assertIn("outcome may be unknown", result.content)

    async def test_non_transient_error_is_not_retried_or_converted(self) -> None:
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            raise RepositoryToolError("invalid path")

        with self.assertRaises(RepositoryToolError):
            await self.middleware.awrap_tool_call(
                build_request("demo_search"),
                handler,
            )

        self.assertEqual(call_count, 1)

    async def test_mcp_business_error_message_is_not_retried(self) -> None:
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return ToolMessage(
                content="server rejected the request",
                tool_call_id="call-1",
                status="error",
            )

        result = await self.middleware.awrap_tool_call(
            build_request("demo_search"),
            handler,
        )

        self.assertEqual(call_count, 1)
        self.assertEqual(result.status, "error")


class MCPRetrySelectionTests(unittest.TestCase):
    def test_no_safe_tools_produces_no_retry_middleware(self) -> None:
        registry = ToolPolicyRegistry(
            {
                "demo_create": build_policy(
                    idempotent=False,
                    side_effect=True,
                )
            }
        )

        middleware = build_mcp_retry_middleware(
            mcp_tools=[SimpleNamespace(name="demo_create")],
            policy_registry=registry,
        )

        self.assertIsNone(middleware)

    def test_all_safe_tools_need_no_no_retry_boundary(self) -> None:
        registry = ToolPolicyRegistry(
            {
                "demo_search": build_policy(
                    idempotent=True,
                    side_effect=False,
                )
            }
        )

        middleware = build_mcp_no_retry_failure_middleware(
            mcp_tools=[SimpleNamespace(name="demo_search")],
            policy_registry=registry,
        )

        self.assertIsNone(middleware)


if __name__ == "__main__":
    unittest.main()
