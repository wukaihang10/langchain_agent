import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "langchain_agent"
FACADE_PATH = PACKAGE_ROOT / "repository_knowledge" / "service.py"
INTERNAL_ROOT = PACKAGE_ROOT / "repository_knowledge" / "_internal"


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)

    return modules


class RepositoryKnowledgeArchitectureTests(unittest.TestCase):
    def test_only_service_facade_imports_private_implementation(self) -> None:
        production_files = [PROJECT_ROOT / "main.py", *PACKAGE_ROOT.rglob("*.py")]
        violations: list[str] = []

        for path in production_files:
            if path == FACADE_PATH or INTERNAL_ROOT in path.parents:
                continue

            for module in imported_modules(path):
                if "repository_knowledge._internal" in module:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")

        self.assertEqual(
            violations,
            [],
            "Private repository-knowledge imports bypassed the service facade:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
