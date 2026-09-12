import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from langsmith import schemas
from langsmith.run_trees import RunTree

from evals.evaluators.trace_evidence import project_tool_evidence

TRACE_ID = UUID("00000000-0000-0000-0000-000000000001")
START_TIME = datetime(2026, 9, 11, tzinfo=UTC)


def make_run(
    name: str,
    *,
    run_type: str = "chain",
    dotted_order: str = "",
    inputs: dict | None = None,
    outputs: dict | None = None,
    error: str | None = None,
    child_runs: list[schemas.Run] | None = None,
    child_run_ids: list[UUID] | None = None,
    start_offset: int = 0,
) -> schemas.Run:
    run_id = uuid4()
    if child_run_ids is None and child_runs is not None:
        child_run_ids = [child.id for child in child_runs]
    return schemas.Run(
        id=run_id,
        trace_id=TRACE_ID,
        name=name,
        run_type=run_type,
        start_time=START_TIME + timedelta(seconds=start_offset),
        dotted_order=dotted_order,
        inputs=inputs or {},
        outputs=outputs,
        error=error,
        child_runs=child_runs,
        child_run_ids=child_run_ids,
    )


class ToolEvidenceProjectionTests(unittest.TestCase):
    def test_projects_live_run_tree_without_declared_child_ids(self):
        tool = RunTree(
            name="read_file",
            run_type="tool",
            inputs={"file_path": "src/harbor_tasks/config.py"},
            outputs={"output": "DEFAULT_MAX_ATTEMPTS = 4"},
        )
        root = RunTree(
            name="repository_fact_target",
            child_runs=[tool],
        )

        projection = project_tool_evidence(root)

        self.assertEqual(
            projection,
            {
                "status": "available",
                "evidence": [
                    {
                        "tool_name": "read_file",
                        "inputs": {
                            "file_path": "src/harbor_tasks/config.py",
                        },
                        "output": "DEFAULT_MAX_ATTEMPTS = 4",
                        "error": None,
                    }
                ],
            },
        )

    def test_recursively_projects_only_tool_runs(self):
        tool = make_run(
            "read_file",
            run_type="tool",
            inputs={"file_path": "src/harbor_tasks/config.py"},
            outputs={"output": "DEFAULT_MAX_ATTEMPTS = 4"},
        )
        middleware = make_run(
            "ToolErrorMiddleware.awrap_tool_call",
            child_runs=[tool],
        )
        model = make_run(
            "ChatDeepSeek",
            run_type="llm",
            inputs={"messages": ["large prompt"]},
            outputs={"generations": ["large response"]},
        )
        root = make_run(
            "repository_fact_target",
            child_runs=[model, middleware],
            child_run_ids=[model.id, middleware.id, tool.id],
        )

        projection = project_tool_evidence(root)

        self.assertEqual(
            projection,
            {
                "status": "available",
                "evidence": [
                    {
                        "tool_name": "read_file",
                        "inputs": {
                            "file_path": "src/harbor_tasks/config.py",
                        },
                        "output": "DEFAULT_MAX_ATTEMPTS = 4",
                        "error": None,
                    }
                ],
            },
        )

    def test_orders_tools_by_dotted_order(self):
        later = make_run(
            "read_file",
            run_type="tool",
            dotted_order="20260911Z02",
            inputs={"file_path": "later.py"},
            outputs={"output": "later"},
        )
        earlier = make_run(
            "search_code",
            run_type="tool",
            dotted_order="20260911Z01",
            inputs={"keyword": "earlier"},
            outputs={"output": "earlier"},
        )
        root = make_run("root", child_runs=[later, earlier])

        projection = project_tool_evidence(root)

        self.assertEqual(
            [item["tool_name"] for item in projection["evidence"]],
            ["search_code", "read_file"],
        )

    def test_normalizes_structured_output_and_run_error(self):
        tool = make_run(
            "search_code",
            run_type="tool",
            inputs={"max_results": 5, "keyword": "retry"},
            outputs={"output": {"matches": ["retry.py:10"]}},
            error="tool transport failed",
        )
        root = make_run("root", child_runs=[tool])

        projection = project_tool_evidence(root)

        self.assertEqual(
            projection["evidence"],
            [
                {
                    "tool_name": "search_code",
                    "inputs": {"max_results": 5, "keyword": "retry"},
                    "output": '{"matches":["retry.py:10"]}',
                    "error": "tool transport failed",
                }
            ],
        )

    def test_distinguishes_unavailable_trace_from_available_empty_trace(self):
        self.assertEqual(
            project_tool_evidence(None),
            {"status": "unavailable", "evidence": []},
        )
        self.assertEqual(
            project_tool_evidence(make_run("root")),
            {"status": "unavailable", "evidence": []},
        )
        self.assertEqual(
            project_tool_evidence(make_run("root", child_runs=[])),
            {"status": "available", "evidence": []},
        )

    def test_marks_projection_partial_when_declared_children_are_missing(self):
        missing_child_id = uuid4()
        partially_loaded = make_run(
            "middleware",
            child_run_ids=[missing_child_id],
        )
        root = make_run("root", child_runs=[partially_loaded])

        projection = project_tool_evidence(root)

        self.assertEqual(projection["status"], "partial")
        self.assertEqual(projection["evidence"], [])

    def test_marks_projection_partial_when_an_output_is_truncated(self):
        tool = make_run(
            "read_file",
            run_type="tool",
            outputs={"output": "x" * 100_000},
        )
        root = make_run("root", child_runs=[tool])

        projection = project_tool_evidence(root)

        self.assertEqual(projection["status"], "partial")
        self.assertLess(len(projection["evidence"][0]["output"]), 100_000)
        self.assertTrue(
            projection["evidence"][0]["output"].endswith("[trace evidence truncated]")
        )

    def test_marks_projection_partial_when_total_budget_is_exhausted(self):
        tools = [
            make_run(
                f"read_file_{index}",
                run_type="tool",
                dotted_order=f"20260911Z{index:02d}",
                outputs={"output": "x" * 11_000},
            )
            for index in range(6)
        ]
        root = make_run("root", child_runs=tools)

        projection = project_tool_evidence(root)

        self.assertEqual(projection["status"], "partial")
        self.assertLess(len(projection["evidence"]), len(tools))

    def test_does_not_mutate_the_run_tree(self):
        tool = make_run(
            "read_file",
            run_type="tool",
            outputs={"output": {"content": "source"}},
        )
        root = make_run("root", child_runs=[tool])
        original_outputs = tool.outputs

        project_tool_evidence(root)

        self.assertIs(tool.outputs, original_outputs)
        self.assertEqual(tool.outputs, {"output": {"content": "source"}})


if __name__ == "__main__":
    unittest.main()
