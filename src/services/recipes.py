"""Portable training recipe helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.services.runs import read_run_record

SCHEMA_VERSION = "localtune.recipe.v1"
TRAINING_KEYS = {
    "mode",
    "max_steps",
    "max_seq_length",
    "lora_r",
    "gradient_accumulation_steps",
    "logging_steps",
    "save_steps",
    "no_fallback",
    "do_eval",
}


def recipes_root(project_root: Path) -> Path:
    return project_root / "outputs" / "localtune-recipes"


def safe_recipe_name(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("Recipe name is required")
    return normalized[:80]


def export_run_recipe(project_root: Path, run_id: str, name: str | None = None) -> dict[str, Any]:
    record = read_run_record(project_root, run_id)
    if not record or record.get("kind") != "training":
        raise ValueError(f"Training run not found: {run_id}")
    params = record.get("params") or {}
    recipe_name = safe_recipe_name(name or f"{record.get('model_id') or 'model'}_{record.get('dataset_profile') or 'dataset'}_{run_id}")
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "name": recipe_name,
        "description": f"Exported from LocalTune training run {run_id}",
        "created_at": datetime.now().isoformat(),
        "model": {
            "id": params.get("model_id") or record.get("model_id"),
            "branch": params.get("branch") or record.get("branch"),
        },
        "dataset": {
            "profile": params.get("dataset_profile") or record.get("dataset_profile"),
        },
        "training": {
            key: value
            for key, value in params.items()
            if key in TRAINING_KEYS and key != "resume_from_checkpoint"
        },
        "source_run": {
            "id": run_id,
            "status": record.get("status"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "elapsed_seconds": record.get("elapsed_seconds"),
        },
    }
    root = recipes_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{recipe_name}.yaml"
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"ok": True, "name": recipe_name, "path": relative_path(project_root, path), "recipe": recipe}


def import_recipe(project_root: Path, value: str) -> dict[str, Any]:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    allowed_roots = [
        recipes_root(project_root).resolve(),
        (project_root / "examples" / "recipes").resolve(),
    ]
    resolved = path.resolve()
    if not any(root in resolved.parents for root in allowed_roots) or not path.is_file():
        raise ValueError("Recipe path is outside the LocalTune recipe library")
    recipe = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if recipe.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported recipe schema")
    model = recipe.get("model") or {}
    dataset = recipe.get("dataset") or {}
    training = recipe.get("training") or {}
    payload = {
        **{key: value for key, value in training.items() if key in TRAINING_KEYS},
        "model_id": model.get("id"),
        "branch": model.get("branch"),
        "dataset_profile": dataset.get("profile"),
    }
    if not payload.get("model_id") or not payload.get("branch") or not payload.get("dataset_profile"):
        raise ValueError("Recipe is missing model, branch, or dataset profile")
    return {"ok": True, "path": relative_path(project_root, path), "recipe": recipe, "payload": payload}


def list_recipes(project_root: Path) -> list[dict[str, Any]]:
    items = []
    roots = [recipes_root(project_root), project_root / "examples" / "recipes"]
    paths = [path for root in roots if root.exists() for path in root.glob("*.yaml")]
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            recipe = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        items.append({
            "name": recipe.get("name") or path.stem,
            "description": recipe.get("description") or "",
            "path": relative_path(project_root, path),
            "created_at": recipe.get("created_at"),
            "model_id": (recipe.get("model") or {}).get("id"),
            "branch": (recipe.get("model") or {}).get("branch"),
            "dataset_profile": (recipe.get("dataset") or {}).get("profile"),
            "source_run_id": (recipe.get("source_run") or {}).get("id"),
        })
    return items


def relative_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)
