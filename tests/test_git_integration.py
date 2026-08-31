import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_agent.integrations.git import (
    collect_file_editions,
    collect_untracked_files,
)


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

    @patch("langchain_agent.integrations.git.subprocess.run")
    def test_git_failure_remains_a_programming_visible_error(self, run):
        run.return_value = SimpleNamespace(
            returncode=1,
            stderr="not a git repository",
            stdout="",
        )

        with self.assertRaisesRegex(RuntimeError, "not a git repository"):
            collect_untracked_files("repository")

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["git", "-C", "repository"])


if __name__ == "__main__":
    unittest.main()
