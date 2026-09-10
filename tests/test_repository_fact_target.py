import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from evals.targets.repository_fact import RepositoryFactTarget
from langchain_agent.app.config import AppConfig, AppPaths
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.permissions.models import PermissionMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingAgent:
    def __init__(self) -> None:
        self.calls = []

    async def ainvoke(self, input_value, *, config, context):
        self.calls.append(
            {
                "input": input_value,
                "config": config,
                "context": context,
            }
        )
        return {
            "messages": [AIMessage(content="grounded answer")],
            "git_audit_status": "available",
            "edited_file_list": [],
        }


class RepositoryFactTargetTests(unittest.IsolatedAsyncioTestCase):
    def build_target(self, agent=None):
        context = AgentContext(
            repository_path=str(PROJECT_ROOT),
            repository_knowledge=object(),
            permission_mode=PermissionMode.READ_ONLY,
        )
        config = AppConfig(
            permission_mode=PermissionMode.READ_ONLY,
            paths=AppPaths.under(PROJECT_ROOT / ".agent" / "eval-test"),
            agent_version="evaluation-test-version",
        )
        return RepositoryFactTarget(
            agent=agent or RecordingAgent(),
            context=context,
            config=config,
        )

    async def test_invokes_agent_with_only_question_and_normalizes_output(self):
        agent = RecordingAgent()
        target = self.build_target(agent)

        output = await target(
            {
                "question": "  Is an unknown tool allowed?  ",
            }
        )

        self.assertEqual(
            agent.calls[0]["input"],
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Is an unknown tool allowed?",
                    }
                ]
            },
        )
        self.assertIs(agent.calls[0]["context"], target.context)
        self.assertEqual(
            agent.calls[0]["config"]["run_name"],
            "repository_fact_target",
        )
        self.assertEqual(
            agent.calls[0]["config"]["tags"],
            ["evaluation", "repository_fact"],
        )
        thread_id = agent.calls[0]["config"]["configurable"]["thread_id"]
        self.assertEqual(
            agent.calls[0]["config"]["metadata"],
            {
                "agent_version": "evaluation-test-version",
                "thread_id": thread_id,
                "fixture_version": "repository_fact_v1",
                "permission_mode": "read_only",
            },
        )
        self.assertEqual(
            output,
            {
                "answer": "grounded answer",
                "git_audit_status": "available",
                "edited_files": [],
            },
        )

    async def test_uses_a_fresh_thread_for_every_invocation(self):
        agent = RecordingAgent()
        target = self.build_target(agent)
        inputs = {
            "question": "How many attempts are allowed?",
        }

        await target(inputs)
        await target(inputs)

        thread_ids = [
            call["config"]["configurable"]["thread_id"] for call in agent.calls
        ]
        self.assertEqual(len(set(thread_ids)), 2)
        self.assertTrue(all(value.startswith("eval-") for value in thread_ids))

    async def test_rejects_reference_data_in_target_inputs(self):
        target = self.build_target()

        with self.assertRaisesRegex(
            ValueError,
            "unexpected repository-fact input fields: reference_outputs",
        ):
            await target(
                {
                    "question": "What is the answer?",
                    "reference_outputs": {"answer": "secret"},
                }
            )

    async def test_rejects_case_id_in_target_inputs(self):
        target = self.build_target()

        with self.assertRaisesRegex(
            ValueError,
            "unexpected repository-fact input fields: case_id",
        ):
            await target(
                {
                    "case_id": "repository_fact_001",
                    "question": "What is the answer?",
                }
            )

    def test_requires_read_only_context(self):
        context = AgentContext(
            repository_path=str(PROJECT_ROOT),
            repository_knowledge=object(),
            permission_mode=PermissionMode.DEFAULT,
        )

        with self.assertRaisesRegex(
            ValueError,
            "repository-fact evaluation requires read-only context",
        ):
            RepositoryFactTarget(
                agent=RecordingAgent(),
                context=context,
                config=AppConfig(),
            )


if __name__ == "__main__":
    unittest.main()
