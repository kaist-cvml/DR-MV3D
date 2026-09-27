"""JSON and JSONL helpers shared by the data-preparation scripts."""

import json
import os
from typing import Any


def ensure_dir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist yet.

    Args:
        dir_path: Directory to create. An empty string is ignored, so this can
            be called with ``os.path.dirname(...)`` of a bare file name.
    """
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Read a JSON Lines file.

    Args:
        file_path: Path to the ``.jsonl`` file.

    Returns:
        One dictionary per non-empty line, in file order.
    """
    items: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], file_path: str) -> None:
    """Write a list of dictionaries as JSON Lines.

    Args:
        items: Records to write, one per line.
        file_path: Output path. Parent directories are created if needed.
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Write an object as pretty-printed JSON.

    Args:
        data: Object to serialise.
        file_path: Output path. Parent directories are created if needed.
        indent: Indentation width passed to :func:`json.dump`.
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
