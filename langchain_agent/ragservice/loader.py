from pathlib import Path
from langgraph_agent.ragservice.models import Document


class TextDocumentLoader:
    """现只加载.md和.txt。"""

    SUPPORTED_SUFFIXES = {".md", ".txt"}

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:

        root = Path(directory).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Knowledge directory does not exist: {root}")

        if not root.is_dir():
            raise NotADirectoryError(f"Knowledge path is not a directory: {root}")

        documents: list[Document] = []

        for file_path in sorted(root.rglob("*")):
            if not self._is_supported_file(file_path):
                continue

            document = self.load_file(
                file_path=file_path,
                root=root,
            )

            if document.content.strip():
                documents.append(document)

        return documents

    def load_file(
        self,
        file_path: str | Path,
        root: str | Path | None = None,
    ) -> Document:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Document does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Document path is not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {path.suffix}")

        content = path.read_text(encoding="utf-8")

        source = self._build_source(
            file_path=path,
            root=Path(root).resolve() if root else None,
        )

        return Document(
            id=source,
            content=content,
            source=source,
            metadata={
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "absolute_path": str(path),
            },
        )

    def _is_supported_file(
        self,
        file_path: Path,
    ) -> bool:
        return (
            file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_SUFFIXES
        )

    @staticmethod
    def _build_source(
        file_path: Path,
        root: Path | None,
    ) -> str:
        if root is None:
            return file_path.name

        return file_path.relative_to(root).as_posix()
