from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from dotenv import load_dotenv
from langsmith import Client, schemas
from langsmith.utils import LangSmithNotFoundError

from evals.datasets.repository_fact import load_repository_fact_dataset
from evals.environments.repository_fact import (
    FIXTURE_VERSION,
    RepositoryFactModels,
    open_repository_fact_environment,
)
from evals.evaluators.repository_fact import evaluate_policy_compliance
from evals.targets.repository_fact import (
    RepositoryFactTarget,
    RepositoryFactTargetOutput,
)
from langchain_agent.app.config import AppConfig, AppPaths
from langchain_agent.integrations.model import create_model
from langchain_agent.repository_knowledge import (
    FallbackQueryExpander,
    IdentityQueryExpander,
    LLMQueryExpander,
)
from langchain_agent.repository_knowledge.embedding import (
    SentenceTransformerEmbeddingClient,
)

DATASET_NAME = "repository-fact-v0"
EXPERIMENT_PREFIX = "repository-fact-baseline"
POLICY_EVALUATOR_VERSION = "policy-compliance-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sync_repository_fact_dataset(client: Client) -> schemas.Dataset:
    """Create or update the versioned LangSmith Dataset without duplicates."""

    examples = load_repository_fact_dataset()
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=(
                "Single-turn, read-only repository-fact evaluation fixture v0."
            ),
        )

    desired_examples = [_remote_example(example) for example in examples]
    desired_ids = [example["id"] for example in desired_examples]
    existing_examples = {
        example.id: example
        for example in client.list_examples(
            dataset_id=dataset.id,
            example_ids=desired_ids,
        )
    }
    examples_to_create = [
        example
        for example in desired_examples
        if example["id"] not in existing_examples
    ]
    examples_to_update = [
        example
        for example in desired_examples
        if example["id"] in existing_examples
        and not _remote_example_matches(existing_examples[example["id"]], example)
    ]

    if examples_to_create:
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples_to_create,
        )
    if examples_to_update:
        client.update_examples(
            dataset_id=dataset.id,
            updates=examples_to_update,
        )
    return dataset


async def run_repository_fact_baseline() -> str:
    """Synchronize the Dataset and run the first policy-scored Experiment."""

    paths = AppPaths.user_default()
    load_dotenv(paths.environment_path, override=False)

    client = Client()
    dataset = sync_repository_fact_dataset(client)
    config = AppConfig(agent_version="langchain-agent-v0")
    models = RepositoryFactModels(
        primary=create_model(thinking=False),
        summary=create_model(thinking=False),
        researcher=create_model(thinking=False),
        reviewer=create_model(thinking=False),
    )
    query_expander = FallbackQueryExpander(
        primary=LLMQueryExpander(model=create_model(thinking=False)),
        fallback=IdentityQueryExpander(),
    )

    async with open_repository_fact_environment(
        models=models,
        embedding_client_factory=lambda: SentenceTransformerEmbeddingClient(
            model_name=config.embedding_model_name,
        ),
        query_expander=query_expander,
        config=config,
    ) as environment:
        results = await client.aevaluate(
            _as_async_target(environment.target),
            data=dataset.id,
            evaluators=[evaluate_policy_compliance],
            metadata=_experiment_metadata(config),
            experiment_prefix=EXPERIMENT_PREFIX,
            description=(
                "First repository-fact baseline with deterministic policy feedback."
            ),
            max_concurrency=1,
            num_repetitions=1,
            error_handling="log",
        )
        await results.wait()

    return results.experiment_name


def _as_async_target(
    target: RepositoryFactTarget,
) -> Callable[
    [Mapping[str, Any]],
    Awaitable[RepositoryFactTargetOutput],
]:
    async def repository_fact_target(
        inputs: Mapping[str, Any],
    ) -> RepositoryFactTargetOutput:
        return await target(inputs)

    return repository_fact_target


def _remote_example(example: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(example["metadata"])
    case_id = metadata["case_id"]
    return {
        "id": _example_id(case_id),
        "inputs": dict(example["inputs"]),
        "outputs": dict(example["reference_outputs"]),
        "metadata": metadata,
    }


def _example_id(case_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"langchain-agent/{DATASET_NAME}/{case_id}")


def _remote_example_matches(
    existing: schemas.Example,
    desired: dict[str, Any],
) -> bool:
    desired_metadata = desired["metadata"]
    existing_metadata = existing.metadata or {}
    return (
        existing.inputs == desired["inputs"]
        and existing.outputs == desired["outputs"]
        and all(
            existing_metadata.get(key) == value
            for key, value in desired_metadata.items()
        )
    )


def _experiment_metadata(config: AppConfig) -> dict[str, Any]:
    return {
        "agent_version": config.agent_version,
        "model_name": os.environ["MODEL_NAME"],
        "model_thinking": False,
        "model_temperature": 0,
        "fixture_version": FIXTURE_VERSION,
        "dataset_name": DATASET_NAME,
        "evaluator_version": POLICY_EVALUATOR_VERSION,
        "permission_mode": "read_only",
        "num_repetitions": 1,
        "execution_environment": "local",
    }


def main() -> None:
    experiment_name = asyncio.run(run_repository_fact_baseline())
    print(json.dumps({"experiment": experiment_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
