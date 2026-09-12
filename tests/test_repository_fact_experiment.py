import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from langsmith.utils import LangSmithNotFoundError

from evals.experiments.repository_fact import (
    DATASET_NAME,
    _as_async_target,
    _experiment_metadata,
    sync_repository_fact_dataset,
)
from langchain_agent.app.config import AppConfig


class RepositoryFactDatasetSyncTests(unittest.TestCase):
    def test_creates_dataset_and_upserts_stable_examples(self):
        dataset = SimpleNamespace(id=uuid4())
        client = Mock()
        client.read_dataset.side_effect = LangSmithNotFoundError("missing")
        client.create_dataset.return_value = dataset
        client.list_examples.side_effect = LangSmithNotFoundError("Examples not found")

        result = sync_repository_fact_dataset(client)

        self.assertIs(result, dataset)
        client.create_dataset.assert_called_once_with(
            DATASET_NAME,
            description=(
                "Single-turn, read-only repository-fact evaluation fixture v0."
            ),
        )
        client.list_examples.assert_not_called()
        uploaded = client.create_examples.call_args.kwargs["examples"]
        self.assertEqual(len(uploaded), 8)
        self.assertEqual(
            [example["metadata"]["case_id"] for example in uploaded],
            [f"repository_fact_{number:03d}" for number in range(1, 9)],
        )
        self.assertEqual(len({example["id"] for example in uploaded}), 8)
        self.assertTrue(
            all(set(example["inputs"]) == {"question"} for example in uploaded)
        )
        self.assertTrue(
            all("required_facts" in example["outputs"] for example in uploaded)
        )

    def test_unchanged_existing_examples_are_not_written_again(self):
        dataset = SimpleNamespace(id=uuid4())
        first_client = Mock()
        first_client.read_dataset.return_value = dataset
        first_client.list_examples.return_value = []
        second_client = Mock()
        second_client.read_dataset.return_value = dataset

        sync_repository_fact_dataset(first_client)
        first_examples = first_client.create_examples.call_args.kwargs["examples"]
        second_client.list_examples.return_value = [
            SimpleNamespace(
                id=example["id"],
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata={**example["metadata"], "dataset_split": ["base"]},
            )
            for example in first_examples
        ]
        sync_repository_fact_dataset(second_client)

        first_client.create_dataset.assert_not_called()
        second_client.create_dataset.assert_not_called()
        second_client.create_examples.assert_not_called()
        second_client.update_examples.assert_not_called()

    def test_populates_an_existing_dataset_when_example_lookup_returns_not_found(
        self,
    ):
        dataset = SimpleNamespace(id=uuid4())
        client = Mock()
        client.read_dataset.return_value = dataset
        client.list_examples.side_effect = LangSmithNotFoundError("Examples not found")

        result = sync_repository_fact_dataset(client)

        self.assertIs(result, dataset)
        uploaded = client.create_examples.call_args.kwargs["examples"]
        self.assertEqual(len(uploaded), 8)
        client.update_examples.assert_not_called()

    def test_updates_only_existing_examples_whose_owned_content_changed(self):
        dataset = SimpleNamespace(id=uuid4())
        initial_client = Mock()
        initial_client.read_dataset.return_value = dataset
        initial_client.list_examples.return_value = []
        sync_repository_fact_dataset(initial_client)
        desired = initial_client.create_examples.call_args.kwargs["examples"]

        existing = [
            SimpleNamespace(
                id=example["id"],
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
            )
            for example in desired
        ]
        existing[0].inputs = {"question": "stale question"}
        update_client = Mock()
        update_client.read_dataset.return_value = dataset
        update_client.list_examples.return_value = existing

        sync_repository_fact_dataset(update_client)

        updates = update_client.update_examples.call_args.kwargs["updates"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["metadata"]["case_id"], "repository_fact_001")
        update_client.create_examples.assert_not_called()


class RecordingTarget:
    def __init__(self):
        self.inputs = []

    async def __call__(self, inputs):
        self.inputs.append(inputs)
        return {"answer": "ok"}


class RepositoryFactTargetAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_callable_instance_as_an_async_function(self):
        target = RecordingTarget()

        adapter = _as_async_target(target)
        output = await adapter({"question": "What is the policy?"})

        self.assertTrue(inspect.iscoroutinefunction(adapter))
        self.assertEqual(target.inputs, [{"question": "What is the policy?"}])
        self.assertEqual(output, {"answer": "ok"})


class RepositoryFactExperimentMetadataTests(unittest.TestCase):
    @patch.dict(os.environ, {"MODEL_NAME": "test-model"})
    def test_records_the_agreed_experiment_dimensions(self):
        metadata = _experiment_metadata(AppConfig(agent_version="test-agent-v7"))

        self.assertEqual(
            metadata,
            {
                "agent_version": "test-agent-v7",
                "model_name": "test-model",
                "model_thinking": False,
                "model_temperature": 0,
                "fixture_version": "repository-fact-v0",
                "dataset_name": DATASET_NAME,
                "evaluator_version": "repository-fact-evaluator-v0",
                "permission_mode": "read_only",
                "num_repetitions": 1,
                "execution_environment": "local",
            },
        )
        self.assertNotIn("dataset_version", metadata)


if __name__ == "__main__":
    unittest.main()
