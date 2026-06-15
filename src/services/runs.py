"""Persistent run metadata for training and inference jobs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "localtune.run.v1"
logger = logging.getLogger(__name__)


def relative_path(project_root: Path, path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(project_root))
    except Exception as exc:
        logger.debug("Failed to make run path relative to %s: %s", project_root, exc, exc_info=True)
        return str(path)


def runs_root(project_root: Path) -> Path:
    return project_root / "outputs" / "localtune-runs"


def run_dir(project_root: Path, run_id: str) -> Path:
    return runs_root(project_root) / run_id


def metadata_path(project_root: Path, run_id: str) -> Path:
    return run_dir(project_root, run_id) / "metadata.json"


def create_run_record(project_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().isoformat()
    run_id = str(record["id"])
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "kind": record.get("kind", "training"),
        "status": record.get("status", "created"),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
        **record,
    }
    _write_record(project_root, run_id, normalized)
    return normalized


def update_run_record(project_root: Path, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    existing = read_run_record(project_root, run_id) or {
        "schema_version": SCHEMA_VERSION,
        "id": run_id,
        "created_at": datetime.now().isoformat(),
    }
    existing.update(updates)
    existing["updated_at"] = datetime.now().isoformat()
    _write_record(project_root, run_id, existing)
    return existing


def read_run_record(project_root: Path, run_id: str) -> dict[str, Any] | None:
    path = metadata_path(project_root, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read run metadata from %s: %s", path, exc, exc_info=True)
        return None


def list_run_records(project_root: Path) -> list[dict[str, Any]]:
    root = runs_root(project_root)
    if not root.exists():
        return []
    records = []
    for path in root.glob("*/metadata.json"):
        record = read_run_record(project_root, path.parent.name)
        if record:
            records.append(record)
    return sorted(records, key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)


def _write_record(project_root: Path, run_id: str, record: dict[str, Any]) -> None:
    path = metadata_path(project_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
