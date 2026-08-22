from __future__ import annotations

import tokenize
from pathlib import Path

from langgraph_agent.ragservice.models import Document


class PythonDocumentLoader:
    """
    加载 Python 项目中的 .py文件。

    默认跳过虚拟环境、缓存、Git 和 构建目录。
    """

    DEFAULT_IGNORED_DIRECTORIES = {
        ".git",
        ".idea",
        ".vscode",
        ".venv",
        ".rag_index",
        "venv",
        "env",
        "__pycache__",
        "site-packages",
        "node_modules",
        "build",
        "dist",
        "tests",
        "workspace",
    }

    def __init__(
        self,
        ignored_directories: set[str] | None = None,
    ) -> None:
        self.ignored_directories = (
            set(ignored_directories)
            if ignored_directories is not None
            else set(self.DEFAULT_IGNORED_DIRECTORIES)
        )

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:
        root = self._resolve_directory(directory)

        documents: list[Document] = []

        for file_path in self.discover_files(root):
            document = self.load_file(
                file_path=file_path,
                root=root,
            )

            if document.content.strip():
                documents.append(document)

        return documents

    def discover_files(
        self,
        directory: str | Path,
    ) -> list[Path]:
        """
        返回当前 Loader 实际会处理的 Python 文件。

        文件变化检测和文档加载共用该方法，
        避免两套扫描规则不一致。
        """

        root = self._resolve_directory(directory)

        files: list[Path] = []

        for file_path in sorted(root.rglob("*.py")):
            if not file_path.is_file():
                continue

            # 第一版不跟随源码文件符号链接，
            # 避免意外索引仓库目录之外的文件。
            if file_path.is_symlink():
                continue

            if self._should_ignore(
                file_path=file_path,
                root=root,
            ):
                continue

            files.append(file_path)

        return files

    def load_file(
        self,
        file_path: str | Path,
        root: str | Path | None = None,
    ) -> Document:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Python file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Python path is not a file: {path}")

        if path.suffix.lower() != ".py":
            raise ValueError(f"Not a Python file: {path}")

        root_path = Path(root).resolve() if root is not None else None

        source = self._build_source(
            file_path=path,
            root=root_path,
        )

        # tokenize.open() 会遵循 Python 文件开头的 encoding 声明，比固定使用 UTF-8 更适合源码。
        with tokenize.open(path) as file:
            content = file.read()

        return Document(
            id=f"python:{source}",
            content=content,
            source=source,
            metadata={
                "file_name": path.name,
                "extension": ".py",
                "absolute_path": str(path),
                "language": "python",
                "source_type": "python_source",
            },
        )

    def _should_ignore(
        self,
        file_path: Path,
        root: Path,
    ) -> bool:
        relative_path = file_path.relative_to(root)

        return any(
            part in self.ignored_directories for part in relative_path.parts[:-1]
        )

    @staticmethod
    def _resolve_directory(
        directory: str | Path,
    ) -> Path:
        root = Path(directory).resolve()

        if not root.exists():
            raise FileNotFoundError("Python project does not exist: " f"{root}")

        if not root.is_dir():
            raise NotADirectoryError(
                "Python project path is not " f"a directory: {root}"
            )

        return root

    @staticmethod
    def _build_source(
        file_path: Path,
        root: Path | None,
    ) -> str:
        if root is None:
            return file_path.name

        return file_path.relative_to(root).as_posix()
