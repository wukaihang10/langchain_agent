from __future__ import annotations

import hashlib
import os
from pathlib import Path


def repository_cache_key(repository_path: str | Path) -> str:
    """Return a stable, filesystem-safe key for one local repository path."""

    path = Path(repository_path).expanduser().resolve()
    normalized_path = os.path.normcase(str(path))
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]

