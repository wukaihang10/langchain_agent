import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "langchain_agent"


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)

    return modules


class PackageArchitectureTests(unittest.TestCase):
    def assert_no_imports(
        self,
        root: Path,
        forbidden_prefixes: tuple[str, ...],
    ) -> None:
        violations: list[str] = []

        for path in root.rglob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} -> {module}"
                    )

        self.assertEqual(
            violations,
            [],
            "Invalid package dependency direction:\n" + "\n".join(violations),
        )

    def test_repository_knowledge_does_not_depend_on_agent_runtime(self):
        self.assert_no_imports(
            PACKAGE_ROOT / "repository_knowledge",
            (
                "langchain_agent.app",
                "langchain_agent.cli",
                "langchain_agent.harness",
                "langchain_agent.tools",
            ),
        )

    def test_harness_does_not_depend_on_cli_or_bootstrap(self):
        self.assert_no_imports(
            PACKAGE_ROOT / "harness",
            (
                "langchain_agent.cli",
                "langchain_agent.app.bootstrap",
            ),
        )

    def test_tools_do_not_depend_on_cli_or_composition_root(self):
        self.assert_no_imports(
            PACKAGE_ROOT / "tools",
            (
                "langchain_agent.cli",
                "langchain_agent.app.bootstrap",
            ),
        )


if __name__ == "__main__":
    unittest.main()
