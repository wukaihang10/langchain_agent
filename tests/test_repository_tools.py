import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_agent.tools.errors import RepositoryToolError
from langchain_agent.tools.repository import (
    list_files,
    read_file,
    replace_in_file,
    search_code,
    summarize_repository,
    write_file,
)


def build_runtime(repository_path: Path):
    return SimpleNamespace(
        context=SimpleNamespace(
            repository_path=str(repository_path),
        )
    )


class RepositoryToolSchemaTests(unittest.TestCase):
    def test_numeric_limits_are_part_of_tool_schema(self) -> None:
        invalid_calls = [
            (list_files, {"max_files": 0}),
            (
                read_file,
                {
                    "file_path": "file.py",
                    "start_line": 0,
                    "start_column": 0,
                    "max_chars": 100,
                },
            ),
            (search_code, {"keyword": "value", "max_results": 0}),
            (summarize_repository, {"readme_max_chars": 99}),
        ]

        for repository_tool, arguments in invalid_calls:
            with (
                self.subTest(tool=repository_tool.name),
                self.assertRaises(ValueError),
            ):
                repository_tool.tool_call_schema.model_validate(arguments)

    def test_required_text_must_not_be_empty(self) -> None:
        invalid_calls = [
            (read_file, {"file_path": ""}),
            (search_code, {"keyword": "   "}),
            (write_file, {"file_path": "", "content": "content"}),
            (
                replace_in_file,
                {
                    "file_path": "file.py",
                    "old_text": "",
                    "new_text": "replacement",
                },
            ),
        ]

        for repository_tool, arguments in invalid_calls:
            with (
                self.subTest(tool=repository_tool.name),
                self.assertRaises(ValueError),
            ):
                repository_tool.tool_call_schema.model_validate(arguments)


class RepositoryToolFailureTests(unittest.TestCase):
    def test_read_file_missing_path_is_expected_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = build_runtime(Path(directory))

            with self.assertRaises(RepositoryToolError):
                read_file.func(
                    runtime=runtime,
                    file_path="missing.py",
                )

    def test_replace_in_file_rejects_ambiguous_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "example.py"
            original_content = "value = 1\nvalue = 1\n"
            path.write_text(original_content, encoding="utf-8")
            runtime = build_runtime(root)

            with self.assertRaises(RepositoryToolError):
                replace_in_file.func(
                    runtime=runtime,
                    file_path="example.py",
                    old_text="value = 1",
                    new_text="value = 2",
                )

            self.assertEqual(path.read_text(encoding="utf-8"), original_content)

    def test_write_file_refuses_to_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "existing.py"
            path.write_text("original\n", encoding="utf-8")
            runtime = build_runtime(root)

            with self.assertRaises(RepositoryToolError):
                write_file.func(
                    runtime=runtime,
                    file_path="existing.py",
                    content="replacement\n",
                )

            self.assertEqual(path.read_text(encoding="utf-8"), "original\n")


class RepositoryToolPartialSuccessTests(unittest.TestCase):
    def test_search_code_reports_skipped_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.py").write_text("target_value = 1\n", encoding="utf-8")
            (root / "binary.py").write_bytes(b"\xff\xfe")
            runtime = build_runtime(root)

            result = search_code.func(
                runtime=runtime,
                keyword="target_value",
                max_results=20,
            )

            self.assertTrue(result["success"])
            self.assertTrue(result["partial"])
            self.assertEqual(result["skipped_file_count"], 1)
            self.assertEqual(len(result["matches"]), 1)

    def test_summarize_repository_reports_unreadable_readme(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_bytes(b"\xff\xfe")
            (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
            runtime = build_runtime(root)

            result = summarize_repository.func(
                runtime=runtime,
                readme_max_chars=2000,
            )

            self.assertTrue(result["success"])
            self.assertTrue(result["partial"])
            self.assertEqual(len(result["warnings"]), 1)
            self.assertIsNone(result["readme_preview"])


if __name__ == "__main__":
    unittest.main()
