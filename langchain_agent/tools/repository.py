import os
from pathlib import Path
from collections import Counter

from langchain.tools import tool, ToolRuntime

from langchain_agent.context import AgentContext
from langchain_agent.permissions.types import ToolCategory, ToolRisk

MAX_LIST_FILES = 2000
MAX_READ_CHARS = 10000
MAX_SEARCH_RESULTS = 100
MIN_README_CHARS = 100
MAX_README_CHARS = 10000

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
}

LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cpp": "C++",
    ".cc": "C++",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".cs": "C#",
    ".sh": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
}

IMPORTANT_FILENAMES = {
    "readme.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    "license",
    "license.md",
    "makefile",
    "main.py",
    "app.py",
    "manage.py",
}


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def iter_repository_files(root: Path):
    for current_dir, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_dir)
        relative_dir = current.relative_to(root)

        dirnames[:] = [
            name
            for name in dirnames
            if not is_ignored(relative_dir / name) and not (current / name).is_symlink()
        ]

        for filename in filenames:
            file = current / filename
            relative_path = file.relative_to(root)

            if is_ignored(relative_path):
                continue

            try:
                file.resolve().relative_to(root)
            except (OSError, ValueError):
                continue

            if file.is_file():
                yield file, relative_path


@tool
def list_files(runtime: ToolRuntime[AgentContext], max_files: int = 200):
    """List files in the current repository.

    Args:
        max_files: Maximum number of file paths to return.
    """

    repo_path = runtime.context.repository_path
    root = Path(repo_path).resolve()

    if not root.exists():
        return {"success": False, "error": f"Path does not exist: {repo_path}"}

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    if not 1 <= max_files <= MAX_LIST_FILES:
        return {
            "success": False,
            "error": f"max_files must be between 1 and {MAX_LIST_FILES}",
        }

    files = []

    for _, relative_path in iter_repository_files(root):
        files.append(relative_path.as_posix())

    files = sorted(files)
    total_files = len(files)
    truncated = False

    if len(files) > max_files:
        files = files[:max_files]
        truncated = True

    return {
        "success": True,
        "repo_path": str(root),
        "files": files,
        "total_files": total_files,
        "returned_files": len(files),
        "truncated": truncated,
    }


@tool
def read_file(
    runtime: ToolRuntime[AgentContext],
    file_path: str,
    start_line: int = 1,
    start_column: int = 0,
    max_chars: int = MAX_READ_CHARS,
) -> dict:
    """Read source text from a known file in the current repository.

    Use this tool when you already know which file is relevant and need
    authoritative source content, exact implementation details, or code to edit.

    Large files may be returned partially. If `truncated` is true, continue from
    `next_start_line` and `next_start_column` instead of reading the same range again.

    Args:
        file_path: File path relative to the repository root.
        start_line: 1-based line number to start reading from.
        start_column: 0-based character offset within start_line.
        max_chars: Maximum number of characters to return.
    """

    if not isinstance(start_line, int):
        raise TypeError("start_line must be an integer")

    if start_line < 1:
        return {
            "success": False,
            "error": "start_line must be >= 1",
        }

    if not isinstance(start_column, int):
        raise TypeError("start_column must be an integer")

    if start_column < 0:
        return {
            "success": False,
            "error": "start_column must be >= 0",
        }

    if not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")

    if not 1 <= max_chars <= MAX_READ_CHARS:
        return {
            "success": False,
            "error": (f"max_chars must be between 1 and " f"{MAX_READ_CHARS}"),
        }

    repo_path = runtime.context.repository_path
    root = Path(repo_path).resolve()

    if not root.exists() or not root.is_dir():
        return {
            "success": False,
            "error": f"Invalid repository path: {repo_path}",
        }

    path = (root / file_path).resolve()

    try:
        path.relative_to(root)
    except ValueError:
        return {
            "success": False,
            "error": "File path must stay inside the repository.",
        }

    if not path.exists():
        return {
            "success": False,
            "error": f"File does not exist: {file_path}",
        }

    if not path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}",
        }

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return {
            "success": False,
            "error": str(error),
        }

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    # Empty file.
    if total_lines == 0:
        if start_line != 1 or start_column != 0:
            return {
                "success": False,
                "error": "Read position is outside the empty file.",
            }

        return {
            "success": True,
            "repo_path": str(root),
            "file_path": path.relative_to(root).as_posix(),
            "content": "",
            "start_line": 1,
            "start_column": 0,
            "end_line": 1,
            "returned_chars": 0,
            "total_lines": 0,
            "truncated": False,
            "next_start_line": None,
            "next_start_column": None,
        }

    if start_line > total_lines:
        return {
            "success": False,
            "error": (f"start_line {start_line} exceeds " f"file length {total_lines}"),
        }

    start_index = start_line - 1

    if start_column > len(lines[start_index]):
        return {
            "success": False,
            "error": (
                f"start_column {start_column} exceeds " f"line {start_line} length"
            ),
        }

    pieces: list[str] = []
    used_chars = 0

    current_index = start_index
    current_column = start_column
    end_line = start_line

    while current_index < total_lines:
        line = lines[current_index]

        if current_column:
            remaining_line = line[current_column:]
        else:
            remaining_line = line

        remaining_budget = max_chars - used_chars

        # Whole next line fits.
        if len(remaining_line) <= remaining_budget:
            pieces.append(remaining_line)
            used_chars += len(remaining_line)

            end_line = current_index + 1
            current_index += 1
            current_column = 0

            if used_chars >= max_chars:
                break

            continue

        # Normally stop before splitting the next source line.
        if pieces:
            break

        # Edge case: one line itself is larger than max_chars.
        pieces.append(remaining_line[:remaining_budget])
        used_chars += remaining_budget

        end_line = current_index + 1
        current_column += remaining_budget
        break

    content = "".join(pieces)

    # Work out the exact continuation cursor.
    if current_index >= total_lines:
        truncated = False
        next_start_line = None
        next_start_column = None

    else:
        truncated = True
        next_start_line = current_index + 1
        next_start_column = current_column

    return {
        "success": True,
        "repo_path": str(root),
        "file_path": path.relative_to(root).as_posix(),
        "content": content,
        "start_line": start_line,
        "start_column": start_column,
        "end_line": end_line,
        "returned_chars": len(content),
        "total_lines": total_lines,
        "truncated": truncated,
        "next_start_line": next_start_line,
        "next_start_column": next_start_column,
    }


@tool
def search_code(
    runtime: ToolRuntime[AgentContext], keyword: str, max_results: int = 20
):
    """Search for a literal keyword in files in the current repository.

    Use this tool when you know an exact or likely identifier, function name,
    class name, configuration value, error text, or other concrete keyword.

    Args:
        keyword: Keyword or identifier to search for.
        max_results: Maximum number of matches to return.
    """

    repo_path = runtime.context.repository_path

    root = Path(repo_path).resolve()

    if not root.exists():
        return {
            "success": False,
            "error": f"Path not found: {repo_path}",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    if not keyword.strip():
        return {
            "success": False,
            "error": "Keyword cannot be empty",
        }

    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        return {
            "success": False,
            "error": (f"max_results must be between 1 and {MAX_SEARCH_RESULTS}"),
        }

    matches = []
    normalized_keyword = keyword.lower()

    for file, relative_path in iter_repository_files(root):
        try:
            content = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        lines = content.splitlines()

        for line_number, line in enumerate(lines, start=1):
            if normalized_keyword not in line.lower():
                continue

            matches.append(
                {
                    "file_path": relative_path.as_posix(),
                    "line": line_number,
                    "content": line.strip(),
                }
            )

            if len(matches) >= max_results:
                return {
                    "success": True,
                    "repo_path": str(root),
                    "keyword": keyword,
                    "matches": matches,
                    "truncated": True,
                }

    return {
        "success": True,
        "repo_path": str(root),
        "keyword": keyword,
        "matches": matches,
        "truncated": False,
    }


@tool
def summarize_repository(
    runtime: ToolRuntime[AgentContext], readme_max_chars: int = 2000
):
    """Get a structural overview of the current repository.

    Args:
        readme_max_chars: Maximum number of README characters to return.
    """

    repo_path = runtime.context.repository_path

    root = Path(repo_path).resolve()

    if not root.exists():
        return {
            "success": False,
            "error": f"Path not found: {repo_path}",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    if not MIN_README_CHARS <= readme_max_chars <= MAX_README_CHARS:
        return {
            "success": False,
            "error": (
                "readme_max_chars must be between "
                f"{MIN_README_CHARS} and {MAX_README_CHARS}"
            ),
        }

    files = []
    language_counts = Counter()
    extension_counts = Counter()
    important_files = []

    for file, relative_path in iter_repository_files(root):
        files.append(relative_path)

        suffix = file.suffix.lower()

        if suffix:
            extension_counts[suffix] += 1

        language = LANGUAGE_BY_EXTENSION.get(suffix)

        if language:
            language_counts[language] += 1

        if file.name.lower() in IMPORTANT_FILENAMES:
            important_files.append(relative_path.as_posix())

    top_level_structure = sorted(
        item.name + ("/" if item.is_dir() else "")
        for item in root.iterdir()
        if item.name not in IGNORED_DIRECTORIES
    )

    readme_preview = None
    readme_path = None

    for candidate_name in ("README.md", "readme.md", "README", "Readme.md"):
        candidate = root / candidate_name

        if candidate.is_file():
            readme_path = candidate
            break

    if readme_path is not None:
        try:
            readme_content = readme_path.read_text(encoding="utf-8")
            readme_preview = readme_content[:readme_max_chars]
        except (UnicodeDecodeError, PermissionError, OSError):
            readme_preview = None

    return {
        "success": True,
        "repo_name": root.name,
        "repo_path": str(root),
        "total_files": len(files),
        "top_level_structure": top_level_structure,
        "languages": dict(language_counts.most_common()),
        "extensions": dict(extension_counts.most_common()),
        "important_files": sorted(important_files),
        "readme_path": (
            readme_path.relative_to(root).as_posix()
            if readme_path is not None
            else None
        ),
        "readme_preview": readme_preview,
    }


@tool
def write_file(
    runtime: ToolRuntime[AgentContext],
    file_path: str,
    content: str,
) -> dict:
    """Create a new UTF-8 text file in the current repository.

    This tool refuses to overwrite an existing file.

    Args:
        file_path: Path relative to the repository root.
        content: Initial complete content of the new file.
    """

    repo_path = runtime.context.repository_path

    root = Path(repo_path).resolve()

    if not root.exists() or not root.is_dir():
        return {
            "success": False,
            "error": f"Invalid repository path: {repo_path}",
        }

    path = (root / file_path).resolve()

    try:
        path.relative_to(root)
    except ValueError:
        return {
            "success": False,
            "error": "File path must stay inside the repository.",
        }

    if path.exists() and not path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}",
        }

    if not path.parent.exists():
        return {
            "success": False,
            "error": f"Parent directory does not exist: {path.parent}",
        }

    try:
        path.write_text(
            content,
            encoding="utf-8",
        )
    except OSError as error:
        return {
            "success": False,
            "error": str(error),
        }

    return {
        "success": True,
        "file_path": path.relative_to(root).as_posix(),
    }


@tool
def replace_in_file(
    runtime: ToolRuntime[AgentContext],
    file_path: str,
    old_text: str,
    new_text: str,
) -> dict:
    """
    Replace old_text with new_text.
    """
    repo_path = runtime.context.repository_path

    root = Path(repo_path).resolve()

    if not root.exists():
        return {
            "success": False,
            "error": f"Repository path not found: {repo_path}",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Repository path is not a directory: {repo_path}",
        }

    if not old_text:
        return {
            "success": False,
            "error": "old_text cannot be empty",
        }

    path = (root / file_path).resolve()

    try:
        path.relative_to(root)
    except ValueError:
        return {
            "success": False,
            "error": "File path must stay inside the repository",
        }

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }

    if not path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}",
        }

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"File is not valid UTF-8 text: {file_path}",
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except OSError as error:
        return {
            "success": False,
            "error": str(error),
        }

    occurrences = content.count(old_text)

    if occurrences == 0:
        return {
            "success": False,
            "error": ("old_text was not found in " f"{file_path}"),
        }

    if occurrences > 1:
        return {
            "success": False,
            "error": (
                f"old_text appears {occurrences} times in "
                f"{file_path}; provide a more specific snippet"
            ),
        }

    updated_content = content.replace(
        old_text,
        new_text,
        1,
    )

    try:
        path.write_text(
            updated_content,
            encoding="utf-8",
        )
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except OSError as error:
        return {
            "success": False,
            "error": str(error),
        }

    return {
        "success": True,
        "file_path": path.relative_to(root).as_posix(),
        "replacements": 1,
    }


list_files.metadata = {
    "category": ToolCategory.READ.value,
    "idempotent": True,
    "side_effect": False,
    "risk": ToolRisk.LOW.value,
}

read_file.metadata = {
    "category": ToolCategory.READ.value,
    "idempotent": True,
    "side_effect": False,
    "risk": ToolRisk.LOW.value,
}

search_code.metadata = {
    "category": ToolCategory.READ.value,
    "idempotent": True,
    "side_effect": False,
    "risk": ToolRisk.LOW.value,
}

summarize_repository.metadata = {
    "category": ToolCategory.READ.value,
    "idempotent": True,
    "side_effect": False,
    "risk": ToolRisk.LOW.value,
}

write_file.metadata = {
    "category": ToolCategory.WRITE.value,
    "idempotent": True,
    "side_effect": True,
    "risk": ToolRisk.MEDIUM.value,
}

replace_in_file.metadata = {
    "category": ToolCategory.WRITE.value,
    "idempotent": False,
    "side_effect": True,
    "risk": ToolRisk.MEDIUM.value,
}

REPOSITORY_TOOLS = [
    list_files,
    read_file,
    search_code,
    summarize_repository,
    write_file,
    replace_in_file,
]
