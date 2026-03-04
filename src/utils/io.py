from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import yaml

# Create directory (and parents) if it does not already exist
def ensure_directory_exists(path: str | Path) -> Path:
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path

# Write object as pretty-printed JSON to disk
def write_json_file(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    ensure_directory_exists(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)

# Write object as YAML to disk
def write_yaml_file(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    ensure_directory_exists(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(data, file_handle, sort_keys=False)

# Read YAML file from disk into dictionary
def read_yaml_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)

# Merge dictionaries in order, where later values override earlier ones
def merge_dictionaries(*dictionaries: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for current_dictionary in dictionaries:
        for key, value in current_dictionary.items():
            merged[key] = value
    return merged

# Append line of text to log file, creating directories as needed
def append_line_to_text_file(path: str | Path, line: str) -> None:
    file_path = Path(path)
    ensure_directory_exists(file_path.parent)
    with file_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(line)
        if not line.endswith("\n"):
            file_handle.write("\n")