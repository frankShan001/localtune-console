"""Artifact manifest helpers for LocalTune outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "localtune.artifact.v1"
MANIFEST_FILE_NAME = "localtune_artifact.json"
MANIFEST_FILE_NAMES = [MANIFEST_FILE_NAME]

logger = logging.getLogger(__name__)


def read_artifact_manifest(path: Path) -> dict[str, Any] | None:
    for name in MANIFEST_FILE_NAMES:
        manifest_path = path / name
        if manifest_path.exists() and manifest_path.is_file():
            try:
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read manifest %s: %s", manifest_path, exc, exc_info=True)
                return {"error": f"Failed to read {name}"}
    return None


def write_artifact_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def inspect_adapter(path: Path) -> dict[str, Any]:
    weights = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    errors = []
    config = {}
    if not weights.is_file():
        errors.append("adapter_model.safetensors is missing")
    if not config_path.is_file():
        errors.append("adapter_config.json is missing")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"adapter_config.json is invalid: {exc}")
    return {
        "ok": not errors,
        "errors": errors,
        "weights": weights.name if weights.is_file() else "",
        "config_file": config_path.name if config_path.is_file() else "",
        "peft_type": config.get("peft_type"),
        "base_model": config.get("base_model_name_or_path"),
        "target_modules": config.get("target_modules") or [],
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }
