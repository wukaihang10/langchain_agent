import argparse
import asyncio
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_agent.context import AgentContext
from langchain_agent.mcp_config import load_mcp_config
from langchain_agent.model import create_model
from langchain_agent.permissions.types import PermissionMode
from langchain_agent.ragservice.repository_manager import RepositoryKnowledgeManager
from langchain_agent.tools.repository import REPOSITORY_TOOLS
from langchain_agent.permissions.registry import build_tool_policy_registry
from langchain_agent.permissions.middleware import (
    build_hitl_middleware,
    PermissionEnforcementMiddleware,
)

MCP_CONFIG_PATH = Path(".agent") / "mcp.json"


NATIVE_TOOLS = [*REPOSITORY_TOOLS]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "repository_path",
        help="Repository used by AgentContext.",
    )

    return parser.parse_args()


def collect_hitl_decisions(interrupts) -> list[dict]:
    decisions: list[dict] = []

    for interrupt_item in interrupts:
        value = interrupt_item.value

        action_requests = value.get(
            "action_requests",
            [],
        )

        review_configs = value.get(
            "review_configs",
            [],
        )

        for action, review_config in zip(
            action_requests,
            review_configs,
            strict=True,
        ):
            print("\n--- Tool approval required ---")

            print(
                "Tool:",
                action["name"],
            )

            print(
                "Arguments:",
                action["arguments"],
            )

            description = action.get("description")

            if description:
                print(
                    "Reason:",
                    description,
                )

            allowed = review_config["allowed_decisions"]

            print(
                "Allowed:",
                ", ".join(allowed),
            )

            while True:
                answer = input("Approve? [y/n]: ").strip().lower()

                if answer in {"y", "yes"} and "approve" in allowed:
                    decisions.append(
                        {
                            "type": "approve",
                        }
                    )
                    break

                if answer in {"n", "no"} and "reject" in allowed:
                    message = input("Optional rejection reason: ").strip()

                    decision = {
                        "type": "reject",
                    }

                    if message:
                        decision["message"] = message

                    decisions.append(decision)

                    break

                print("Invalid decision. " f"Allowed: {allowed}")

    return decisions


async def invoke_with_hitl(*, agent, input_value, config, context):
    current_input = input_value

    while True:
        result = await agent.ainvoke(
            current_input,
            config=config,
            context=context,
            version="v2",
        )

        if not result.interrupts:
            return result.value

        decisions = collect_hitl_decisions(result.interrupts)

        current_input = Command(
            resume={
                "decisions": decisions,
            }
        )


async def main():
    args = parse_args()

    repository_path = Path(args.repository_path).expanduser().resolve()

    if not repository_path.is_dir():
        raise ValueError(f"Repository does not exist: " f"{repository_path}")

    #
    # Runtime Context
    #

    rag_manager = RepositoryKnowledgeManager(
        retrieval_mode="fast",
    )

    context = AgentContext(
        repository_path=str(repository_path),
        rag_manager=rag_manager,
        permission_mode=PermissionMode.DEFAULT,
    )

    #
    # Tool providers
    #

    mcp_config = load_mcp_config(MCP_CONFIG_PATH)

    if not mcp_config.servers:
        mcp_tools = []

    client = MultiServerMCPClient(
        mcp_config.servers,
        tool_name_prefix=True,
    )

    mcp_tools = await client.get_tools()

    tools = [
        *NATIVE_TOOLS,
        *mcp_tools,
    ]

    policy_registry = build_tool_policy_registry(
        local_tools=NATIVE_TOOLS,
        external_policy_overrides=(mcp_config.tool_policies),
    )

    print("\n--- Tools ---")

    for current_tool in tools:
        print(f"- {current_tool.name}")

    #
    # Create Agent graph
    #

    protected_tool_names = {tool.name for tool in tools}

    hitl_middleware = build_hitl_middleware(
        tools=tools,
        registry=policy_registry,
    )

    permission_enforcement_middleware = PermissionEnforcementMiddleware(
        registry=policy_registry,
        protected_tool_names=protected_tool_names,
    )

    model = create_model()
    summary_model = create_model()

    middleware = [
        hitl_middleware,
        TodoListMiddleware(),
        SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", 30_000),
            keep=("tokens", 8_000),
        ),
        permission_enforcement_middleware,
    ]

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        context_schema=AgentContext,
        system_prompt=(
            "You are a repository analysis agent. "
            "Use repository tools when source-code "
            "information is required. "
            "Do not invent file contents."
        ),
        checkpointer=checkpointer,
        name="create-agent-runtime-demo",
    )

    #
    # Agent loop
    #

    print(
        "\nRepository:",
        context.repository_path,
    )

    config = {"configurable": {"thread_id": "demo"}}

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if not user_input:
            continue

        if user_input in {
            "/exit",
            "/quit",
        }:
            break

        result = await invoke_with_hitl(
            agent=agent,
            input_value={
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            },
            context=context,
            config=config,
        )

        print(
            "\nAgent>",
            result["messages"][-1].content,
        )

        todos = result.get("todos", [])

        if todos:
            print("\n--- Todos ---")

            for index, todo in enumerate(
                todos,
                start=1,
            ):
                print(f"{index}. " f"[{todo['status']}] " f"{todo['content']}")


if __name__ == "__main__":
    asyncio.run(main())
