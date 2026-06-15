"""Unified dataset registry for corpus files and training profiles."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CORPUS_EXTENSIONS = {".jsonl", ".json", ".txt", ".md", ".csv", ".tsv"}
SCHEMA_VERSION = "localtune.dataset_registry.v1"
logger = logging.getLogger(__name__)


@dataclass
class DatasetFile:
    id: str
    name: str
    path: str
    folder: str
    extension: str
    size_bytes: int
    rows: int | None
    updated: str
    inferred_format: str
    task_type: str
    trainable: bool
    role: str = "library"


@dataclass
class DatasetProfile:
    id: str
    name: str
    description: str
    task_type: str
    format: str
    files: dict[str, DatasetFile | None] = field(default_factory=dict)
    total_rows: int = 0
    total_size_bytes: int = 0


def build_dataset_registry(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {}) or {}
    library_files = scan_dataset_files(project_root)
    trainable_files = [item for item in library_files if item.trainable]
    material_files = [item for item in library_files if not item.trainable]
    profiles = build_dataset_profiles(project_root, data_config)
    return {
        "schema_version": SCHEMA_VERSION,
        "root": relative_path(project_root, project_root / "data"),
        "task_type": data_config.get("task_type", "chatml"),
        "format": data_config.get("dataset_format", "chatml_source"),
        "profiles": [asdict(profile) for profile in profiles],
        "files": [asdict(item) for item in trainable_files],
        "materials": [asdict(item) for item in material_files],
        "summary": {
            "profile_count": len(profiles),
            "file_count": len(trainable_files),
            "material_count": len(material_files),
            "total_file_count": len(library_files),
            "total_rows": sum(item.rows or 0 for item in trainable_files),
            "total_size_bytes": sum(item.size_bytes for item in trainable_files),
            "material_size_bytes": sum(item.size_bytes for item in material_files),
        },
    }


def scan_dataset_files(project_root: Path) -> list[DatasetFile]:
    data_dir = project_root / "data"
    if not data_dir.exists():
        return []
    files = []
    for path in sorted(data_dir.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in CORPUS_EXTENSIONS:
            continue
        files.append(dataset_file_from_path(project_root, path, role="library"))
    return files


def build_dataset_profiles(project_root: Path, data_config: dict[str, Any]) -> list[DatasetProfile]:
    profiles_config = data_config.get("profiles", {}) or {}

    profiles = []
    for profile_id, raw_profile in profiles_config.items():
        profile = raw_profile or {}
        files = {
            "train": dataset_file_from_config(project_root, profile.get("train_file"), "train"),
            "val": dataset_file_from_config(project_root, profile.get("val_file"), "val"),
            "test": dataset_file_from_config(project_root, profile.get("test_file"), "test"),
        }
        present_files = [item for item in files.values() if item]
        profiles.append(DatasetProfile(
            id=profile_id,
            name=profile.get("name") or profile_id,
            description=profile.get("description", ""),
            task_type=profile.get("task_type") or data_config.get("task_type", "chatml"),
            format=profile.get("format") or data_config.get("dataset_format", "chatml_source"),
            files=files,
            total_rows=sum(item.rows or 0 for item in present_files),
            total_size_bytes=sum(item.size_bytes for item in present_files),
        ))
    return profiles


def dataset_file_from_config(project_root: Path, value: str | None, role: str) -> DatasetFile | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists() or not path.is_file():
        return DatasetFile(
            id=f"{role}:{value}",
            name=Path(value).name,
            path=str(value),
            folder=str(Path(value).parent),
            extension=Path(value).suffix.lower(),
            size_bytes=0,
            rows=None,
            updated="",
            inferred_format="missing",
            task_type="unknown",
            trainable=False,
            role=role,
        )
    return dataset_file_from_path(project_root, path, role)


def dataset_file_from_path(project_root: Path, path: Path, role: str = "library") -> DatasetFile:
    inferred = infer_dataset_file(project_root, path)
    rel_path = relative_path(project_root, path)
    return DatasetFile(
        id=f"{role}:{rel_path}",
        name=path.name,
        path=rel_path,
        folder=relative_path(project_root, path.parent),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        rows=count_rows(path),
        updated=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        inferred_format=inferred["format"],
        task_type=inferred["task_type"],
        trainable=inferred["trainable"],
        role=role,
    )


def infer_dataset_file(project_root: Path, path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        row = first_jsonl_object(path)
        if row:
            normalized = normalize_row(row)
            return {
                "format": normalized["format"],
                "task_type": normalized["task_type"],
                "trainable": normalized["trainable"],
            }
        return {"format": "jsonl", "task_type": "unknown", "trainable": False}
    if suffix == ".json":
        return {"format": "json", "task_type": "unknown", "trainable": False}
    if suffix in {".txt", ".md"}:
        return {"format": "raw_text", "task_type": "raw_text", "trainable": False}
    if suffix in {".csv", ".tsv"}:
        return {"format": suffix[1:], "task_type": "tabular", "trainable": False}
    return {"format": suffix[1:] or "unknown", "task_type": "unknown", "trainable": False}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task_type = row.get("task_type") or metadata.get("task_type") or "chatml"
    input_text = str(row.get("input") or row.get("source") or row.get("question") or row.get("text") or "")
    output_text = str(row.get("output") or row.get("target") or row.get("answer") or row.get("assistant") or "")
    instruction = str(row.get("instruction") or row.get("prompt") or "")
    if "user" in row and "assistant" in row:
        input_text = str(row.get("user") or "")
        output_text = str(row.get("assistant") or "")
    if row.get("messages") and isinstance(row["messages"], list):
        input_text = next((str(m.get("content", "")) for m in row["messages"] if isinstance(m, dict) and m.get("role") == "user"), "")
        output_text = next((str(m.get("content", "")) for m in reversed(row["messages"]) if isinstance(m, dict) and m.get("role") == "assistant"), "")
    return {
        "format": "localtune_v1" if (instruction or input_text) and output_text else "jsonl",
        "task_type": task_type,
        "trainable": bool((instruction or input_text) and output_text),
    }


def first_jsonl_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read first JSONL object from %s: %s", path, exc, exc_info=True)
        return None
    return None


def count_rows(path: Path) -> int | None:
    if path.suffix.lower() not in {".jsonl", ".csv", ".tsv"}:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        logger.debug("Failed to count rows in %s: %s", path, exc, exc_info=True)
        return None


def relative_path(project_root: Path, path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(project_root))
    except Exception as exc:
        logger.debug("Failed to make path relative to %s: %s", project_root, exc, exc_info=True)
        return str(path)
