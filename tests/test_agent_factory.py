import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_agent.app.agent import NATIVE_TOOLS, build_agent
from langchain_agent.app.config import AppConfig
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.permissions.models import (
    ToolCategory,
    ToolPolicy,
    ToolRisk,
)
from langchain_agent.harness.permissions.registry import (
    build_tool_policy_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@tool
def safe_mcp_tool(query: str) -> str:
    """Return a deterministic MCP search result."""

    return query


@tool
def unknown_mcp_tool(query: str) -> str:
    """Represent an MCP tool without an explicit policy."""

    return query


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class AgentFactoryTests(unittest.IsolatedAsyncioTestCase):
    def test_factory_accepts_mixed_mcp_retry_policies(self):
        models = [
            ToolCallingFakeModel(
                responses=[AIMessage(content="factory response")],
            )
            for _ in range(4)
        ]
        policy_registry = build_tool_policy_registry(
            local_tools=NATIVE_TOOLS,
            external_policy_overrides={
                safe_mcp_tool.name: ToolPolicy(
                    category=ToolCategory.READ,
                    idempotent=True,
                    side_effect=False,
                    risk=ToolRisk.LOW,
                )
            },
        )

        agent = build_agent(
            model=models[0],
            summary_model=models[1],
            researcher_model=models[2],
            reviewer_model=models[3],
            native_tools=NATIVE_TOOLS,
            mcp_tools=[safe_mcp_tool, unknown_mcp_tool],
            policy_registry=policy_registry,
            checkpointer=InMemorySaver(),
            config=AppConfig(),
        )

        self.assertIsNotNone(agent)

    async def test_factory_builds_runnable_agent_from_injected_dependencies(self):
        models = [
            ToolCallingFakeModel(
                responses=[AIMessage(content="factory response")],
            )
            for _ in range(4)
        ]
        agent = build_agent(
            model=models[0],
            summary_model=models[1],
            researcher_model=models[2],
            reviewer_model=models[3],
            native_tools=NATIVE_TOOLS,
            mcp_tools=[],
            policy_registry=build_tool_policy_registry(
                local_tools=NATIVE_TOOLS,
            ),
            checkpointer=InMemorySaver(),
            config=AppConfig(),
        )
        context = AgentContext(
            repository_path=str(PROJECT_ROOT),
            repository_knowledge=object(),
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                    }
                ]
            },
            config={"configurable": {"thread_id": "factory-test"}},
            context=context,
        )

        self.assertEqual(result["messages"][-1].content, "factory response")
        self.assertEqual(result["git_audit_status"], "available")


if __name__ == "__main__":
    unittest.main()
