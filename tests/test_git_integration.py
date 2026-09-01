import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_agent.integrations.git import (
    GitCommandError,
    GitIntegrationError,
    collect_file_editions,
    collect_untracked_files,
)
from langchain_agent.harness.middleware.git_audit import GitAuditMiddleware
from langchain_agent.tools.git_diff import get_git_diff
from langchain_agent.tools.repository_errors import RepositoryToolError


class GitIntegrationTests(unittest.TestCase):
    @patch("langchain_agent.integrations.git._run_git")
    def test_porcelain_rename_keeps_source_and_target_paths(self, run_git):
        run_git.return_value = "R  new.py\0old.py\0"

        result = collect_file_editions("repository")

        self.assertEqual(result["edited_file_list"], ["new.py"])
        self.assertEqual(
            result["edition_list"],
            [
                {
                    "file_path": "new.py",
                    "change_type": "renamed",
                    "old_path": "old.py",
                }
            ],
        )

    @patch("langchain_agent.integrations.git._run_git")
    def test_untracked_files_are_returned_line_by_line(self, run_git):
        run_git.return_value = "a.py\nnested/b.py\n"

        self.assertEqual(
            collect_untracked_files("repository"),
            ["a.py", "nested/b.py"],
        )

    @patch("langchain_agent.integrations.git._run_git")
    def test_clean_worktree_returns_complete_empty_audit(self, run_git):
        run_git.return_value = ""

        self.assertEqual(
            collect_file_editions("repository"),
            {
                "edited_file_list": [],
                "edition_list": [],
            },
        )

    @patch("langchain_agent.integrations.git.subprocess.run")
    def test_git_nonzero_exit_becomes_git_command_error(self, run):
        run.return_value = SimpleNamespace(
            returncode=1,
            stderr="not a git repository",
            stdout="",
        )

        with self.assertRaisesRegex(GitCommandError, "not a git repository"):
            collect_untracked_files("repository")

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["git", "-C", "repository"])

    @patch("langchain_agent.integrations.git.subprocess.run")
    def test_git_process_start_failure_becomes_integration_error(self, run):
        expected_error = FileNotFoundError("git executable was not found")
        run.side_effect = expected_error

        with self.assertRaises(GitIntegrationError) as raised:
            collect_untracked_files("repository")

        self.assertIs(raised.exception.__cause__, expected_error)

    @patch("langchain_agent.integrations.git.subprocess.run")
    def test_unexpected_programming_error_is_not_wrapped(self, run):
        expected_error = TypeError("invalid command construction")
        run.side_effect = expected_error

        with self.assertRaises(TypeError) as raised:
            collect_untracked_files("repository")

        self.assertIs(raised.exception, expected_error)


class GitToolTests(unittest.TestCase):
    @patch("langchain_agent.tools.git_diff.collect_git_diff")
    def test_expected_git_error_becomes_repository_tool_error(self, collect_diff):
        expected_error = GitCommandError("not a git repository")
        collect_diff.side_effect = expected_error
        runtime = SimpleNamespace(
            context=SimpleNamespace(repository_path="repository")
        )

        with self.assertRaises(RepositoryToolError) as raised:
            get_git_diff.func(runtime=runtime, file_path=None)

        self.assertIs(raised.exception.__cause__, expected_error)
        self.assertIn("not a git repository", str(raised.exception))

    @patch("langchain_agent.tools.git_diff.collect_git_diff")
    def test_unexpected_git_tool_error_is_not_wrapped(self, collect_diff):
        expected_error = KeyError("diff")
        collect_diff.side_effect = expected_error
        runtime = SimpleNamespace(
            context=SimpleNamespace(repository_path="repository")
        )

        with self.assertRaises(KeyError) as raised:
            get_git_diff.func(runtime=runtime, file_path=None)

        self.assertIs(raised.exception, expected_error)


class GitAuditMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = GitAuditMiddleware()
        self.runtime = SimpleNamespace(
            context=SimpleNamespace(repository_path="repository")
        )

    @patch("langchain_agent.harness.middleware.git_audit.collect_file_editions")
    def test_success_records_available_audit(self, collect_editions):
        collect_editions.return_value = {
            "edited_file_list": ["file.py"],
            "edition_list": [
                {
                    "file_path": "file.py",
                    "change_type": "modified",
                }
            ],
        }

        result = self.middleware.after_agent({}, self.runtime)

        self.assertEqual(result["git_audit_status"], "available")
        self.assertIsNone(result["git_audit_error"])
        self.assertEqual(result["edited_file_list"], ["file.py"])

    @patch("langchain_agent.harness.middleware.git_audit.collect_file_editions")
    def test_expected_git_error_records_unavailable_audit(self, collect_editions):
        collect_editions.side_effect = GitIntegrationError("Git is unavailable")

        result = self.middleware.after_agent({}, self.runtime)

        self.assertEqual(
            result,
            {
                "edited_file_list": [],
                "edition_list": [],
                "git_audit_status": "unavailable",
                "git_audit_error": "Git is unavailable",
            },
        )

    @patch("langchain_agent.harness.middleware.git_audit.collect_file_editions")
    def test_unexpected_audit_error_is_not_hidden(self, collect_editions):
        collect_editions.side_effect = KeyError("audit")

        with self.assertRaises(KeyError):
            self.middleware.after_agent({}, self.runtime)


if __name__ == "__main__":
    unittest.main()
