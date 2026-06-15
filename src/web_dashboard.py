#!/usr/bin/env python
"""Local web console for starting and monitoring QLoRA training."""

import json
import importlib
import importlib.metadata
import atexit
import io
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from functools import wraps
from pathlib import Path

import yaml
import psutil
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

from scripts.validate_data import validate_file, validate_profile
from src.constants import (
    DEFAULT_DASHBOARD_HOST,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_OUTPUT_DIR,
)
from src.services.artifacts import inspect_adapter, read_artifact_manifest, write_artifact_manifest
from src.services.config_store import ProjectConfigStore
from src.services.datasets import build_dataset_registry, scan_dataset_files
from src.services.errors import LocalTuneError, error_details
from src.services.recipes import export_run_recipe, import_recipe, list_recipes
from src.services.runs import create_run_record, list_run_records, read_run_record, run_dir, update_run_record
from src.services.training_jobs import TrainingManager

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_FALLBACK_DIST_DIR = PROJECT_ROOT / ".tmp" / "frontend-dist"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNTIME_CONFIG_DIR = PROJECT_ROOT / "configs" / "runtime"
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
RUNTIME_CONFIG_KEEP = 10
MODEL_DOWNLOAD_STATE_PATH = RUNTIME_CONFIG_DIR / "model_downloads.json"
MODEL_EXPORT_STATE_PATH = RUNTIME_CONFIG_DIR / "model_exports.json"

logger = logging.getLogger(__name__)

CONFIG_STORE = ProjectConfigStore(BASE_CONFIG_PATH)
_GOLDEN_PATH_CACHE = {}
_GOLDEN_PATH_CACHE_LOCK = threading.Lock()
_GOLDEN_PATH_WARM_STARTED = False
_MODEL_DOWNLOAD_LOCK = threading.Lock()
_MODEL_DOWNLOAD_PROCESSES = {}
_MODEL_EXPORT_LOCK = threading.Lock()
_MODEL_EXPORT_PROCESSES = {}

DOWNLOADABLE_MODEL_CATALOG = [
    {
        "id": "qwen3_5_4b",
        "family": "Qwen",
        "name": "Qwen3.5 4B",
        "provider_name": "Qwen/Qwen3.5-4B",
        "params_b": 4,
        "min_vram_gb": 6,
        "recommended_vram_gb": 10,
        "language_fit": "zh_en",
        "summary": "Light 2026 Qwen model for first local fine-tuning checks.",
        "summary_zh": "适合首次本地微调检查的 2026 年轻量 Qwen 模型。",
    },
    {
        "id": "gemma4_e4b",
        "family": "Gemma",
        "name": "Gemma 4 E4B IT",
        "provider_name": "google/gemma-4-E4B-it",
        "params_b": 4,
        "min_vram_gb": 8,
        "recommended_vram_gb": 12,
        "language_fit": "en_multi",
        "summary": "Very light Gemma 4 instruction model for quick English and multilingual checks.",
        "summary_zh": "很轻量的 Gemma 4 指令模型，适合快速验证英文和多语种链路。",
    },
    {
        "id": "qwen3_5_9b",
        "family": "Qwen",
        "name": "Qwen3.5 9B",
        "provider_name": "Qwen/Qwen3.5-9B",
        "params_b": 9,
        "min_vram_gb": 12,
        "recommended_vram_gb": 16,
        "language_fit": "zh_en",
        "summary": "Current-generation mid-size Qwen model for Chinese-heavy local fine-tuning.",
        "summary_zh": "适合中文优先场景的 2026 年中等规格 Qwen 模型。",
    },
    {
        "id": "gemma4_12b",
        "family": "Gemma",
        "name": "Gemma 4 12B",
        "provider_name": "google/gemma-4-12B",
        "params_b": 12,
        "min_vram_gb": 16,
        "recommended_vram_gb": 20,
        "language_fit": "en_multi",
        "summary": "Good mid-size Gemma 4 model when English or multilingual behavior matters.",
        "summary_zh": "适合英文或多语种效果验证的中等规格 Gemma 4 模型。",
    },
    {
        "id": "qwen3_6_27b",
        "family": "Qwen",
        "name": "Qwen3.6 27B",
        "provider_name": "Qwen/Qwen3.6-27B",
        "params_b": 27,
        "min_vram_gb": 22,
        "recommended_vram_gb": 32,
        "language_fit": "zh_en",
        "summary": "Large Qwen3.6 model for quality-oriented runs; use after a successful test run.",
        "summary_zh": "质量优先的大规格 Qwen3.6 模型，建议先跑通试运行再使用。",
    },
    {
        "id": "qwen3_6_35b_a3b",
        "family": "Qwen",
        "name": "Qwen3.6 35B-A3B",
        "provider_name": "Qwen/Qwen3.6-35B-A3B",
        "params_b": 35,
        "min_vram_gb": 22,
        "recommended_vram_gb": 32,
        "language_fit": "zh_en",
        "summary": "MoE Qwen3.6 option for careful experiments on 24GB-class GPUs.",
        "summary_zh": "面向谨慎实验的 Qwen3.6 MoE 规格，24GB 级显卡建议先试运行。",
    },
    {
        "id": "gemma4_26b_a4b",
        "family": "Gemma",
        "name": "Gemma 4 26B A4B IT",
        "provider_name": "google/gemma-4-26B-A4B-it",
        "params_b": 26,
        "min_vram_gb": 22,
        "recommended_vram_gb": 32,
        "language_fit": "en_multi",
        "summary": "Larger Gemma 4 option; treat as a cautious choice on 24GB-class GPUs.",
        "summary_zh": "较大的 Gemma 4 规格，24GB 级显卡上应谨慎使用。",
    },
]

def _cached_golden_check(key, ttl_seconds, loader):
    now = time.monotonic()
    with _GOLDEN_PATH_CACHE_LOCK:
        cached = _GOLDEN_PATH_CACHE.get(key)
        if cached and now - cached["time"] < ttl_seconds:
            return cached["value"]
    value = loader()
    with _GOLDEN_PATH_CACHE_LOCK:
        _GOLDEN_PATH_CACHE[key] = {"time": now, "value": value}
    return value


def _warm_golden_path_cache():
    for key, loader in (("dependencies", get_environment_dependencies), ("gpu", get_gpu_info)):
        try:
            value = loader()
            with _GOLDEN_PATH_CACHE_LOCK:
                _GOLDEN_PATH_CACHE[key] = {"time": time.monotonic(), "value": value}
        except Exception as exc:
            logger.debug("Unable to warm golden path cache for %s: %s", key, exc, exc_info=True)


def start_golden_path_cache_warmup():
    global _GOLDEN_PATH_WARM_STARTED
    with _GOLDEN_PATH_CACHE_LOCK:
        if _GOLDEN_PATH_WARM_STARTED:
            return
        _GOLDEN_PATH_WARM_STARTED = True
    threading.Thread(target=_warm_golden_path_cache, name="golden-path-cache-warmup", daemon=True).start()


def environment_int(name, default):
    value = str(os.environ.get(name, "")).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer environment variable %s=%r", name, value)
        return default


def load_monitoring_config():
    try:
        config = load_project_config()
        monitoring = config.get("monitoring", {})
        return environment_int(
            "LOCALTUNE_PORT",
            monitoring.get("dashboard_port", DEFAULT_DASHBOARD_PORT),
        )
    except Exception as exc:
        logger.warning("Failed to load monitoring config from %s: %s", BASE_CONFIG_PATH, exc, exc_info=True)
        return environment_int("LOCALTUNE_PORT", DEFAULT_DASHBOARD_PORT)


def load_project_config():
    return CONFIG_STORE.read()


def save_project_config(config):
    CONFIG_STORE.write(config)


def serialized_config_update(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with CONFIG_STORE.lock:
            return function(*args, **kwargs)
    return wrapper


def api_error_response(error, status=None, include_ok=False, default_code="REQUEST_FAILED"):
    code, message, error_status = error_details(error, default_code=default_code)
    payload = {"code": code, "error": message}
    if include_ok:
        payload["ok"] = False
    return jsonify(payload), status or error_status


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if not request.path.startswith("/api/"):
        return error
    if isinstance(error, HTTPException):
        return jsonify({
            "code": f"HTTP_{error.code}",
            "error": error.description,
        }), error.code
    logger.exception("Unhandled API error on %s", request.path)
    return api_error_response(error)


def get_latest_log_file(kind=None):
    active = training_manager.active_log_file()
    if active:
        return active
    if not LOGS_DIR.exists():
        return None
    if kind == "training":
        log_files = list(LOGS_DIR.glob("web_train_*.log"))
    elif kind == "inference":
        log_files = list(LOGS_DIR.glob("inference_run_*.log"))
    else:
        log_files = list(LOGS_DIR.glob("*.log"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)


def resolve_project_path(value):
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def require_project_path(value):
    path = resolve_project_path(value)
    if not path:
        raise ValueError("Missing path")
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError("Path must be inside the project")
    return path


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception as exc:
        logger.debug("Failed to make path relative to project: %s", exc, exc_info=True)
        return str(path)


training_manager = TrainingManager(
    PROJECT_ROOT,
    BASE_CONFIG_PATH,
    RUNTIME_CONFIG_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    resolve_project_path,
    relative_path,
    RUNTIME_CONFIG_KEEP,
)
atexit.register(training_manager.shutdown)


def get_frontend_dist_dir():
    if (FRONTEND_DIST_DIR / "index.html").exists():
        return FRONTEND_DIST_DIR
    if (FRONTEND_FALLBACK_DIST_DIR / "index.html").exists():
        return FRONTEND_FALLBACK_DIST_DIR
    return FRONTEND_DIST_DIR


def send_frontend_entry(directory, filename="index.html"):
    response = send_from_directory(directory, filename)
    response.headers["Cache-Control"] = "no-store"
    return response


def count_jsonl_rows(path):
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception as exc:
        logger.debug("Failed to count JSONL rows in %s: %s", path, exc, exc_info=True)
        return None


def file_info(value):
    path = resolve_project_path(value)
    exists = bool(path and path.exists())
    return {
        "path": relative_path(path) if path else value,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "rows": count_jsonl_rows(path) if exists and path.suffix.lower() == ".jsonl" else None,
        "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
    }


CORPUS_EXTENSIONS = {".jsonl", ".json", ".txt", ".md", ".csv", ".tsv"}
LOCAL_TUNE_REWRITE_MARKER = "请把下面文章改写成上述风格："


def list_corpus_library():
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return {"root": relative_path(data_dir), "groups": [], "files": [], "materials": []}

    all_items = [asdict(item) for item in scan_dataset_files(PROJECT_ROOT)]
    trainable_items = [item for item in all_items if item.get("trainable")]
    material_items = [item for item in all_items if not item.get("trainable")]
    groups = []
    for group_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        folder = relative_path(group_dir)
        files = [item for item in trainable_items if item.get("folder", "").startswith(folder)]
        groups.append({
            "id": group_dir.name,
            "path": folder,
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        })
    return {
        "root": relative_path(data_dir),
        "groups": groups,
        "files": trainable_items,
        "materials": material_items,
    }


def corpus_file_info(path):
    return {
        "name": path.name,
        "path": relative_path(path),
        "folder": relative_path(path.parent),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "rows": count_jsonl_rows(path) if path.suffix.lower() == ".jsonl" else None,
        "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def read_corpus_preview(path, limit=20, offset=0, query=""):
    path = require_project_path(path)
    if not path.exists() or not path.is_file():
        raise ValueError("Corpus file does not exist")
    suffix = path.suffix.lower()
    limit = max(1, min(int(limit or 20), 100))
    offset = max(0, int(offset or 0))
    query = str(query or "").strip().lower()
    info = corpus_file_info(path)
    samples = []
    errors = []
    matched = 0

    def matches(value):
        if not query:
            return True
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False)
        return query in text.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    sample = {"line": line_no, "row": row, "normalized": normalize_corpus_row(row)}
                except json.JSONDecodeError as exc:
                    sample = {"line": line_no, "error": str(exc), "text": line[:500]}
                    errors.append(f"line {line_no}: {exc}")
                if not matches(sample):
                    continue
                if matched >= offset and len(samples) < limit:
                    samples.append(sample)
                matched += 1
    elif suffix == ".json":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            data = json.load(handle)
        rows = data if isinstance(data, list) else [data]
        filtered_rows = [row for row in rows if matches(row)]
        matched = len(filtered_rows)
        samples = [
            {"line": index + 1, "row": row, "normalized": normalize_corpus_row(row) if isinstance(row, dict) else None}
            for index, row in enumerate(filtered_rows[offset:offset + limit], offset)
        ]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if matches(text):
            matched = 1
            if offset == 0:
                samples = [{
                    "line": 1,
                    "text": text[:12000],
                    "normalized": {
                        "format": "raw_text",
                        "task_type": "raw_text",
                        "instruction": "Raw source text. Convert it to LocalTune JSONL before training.",
                        "input": text[:12000],
                        "output": "",
                        "metadata": {"source_file": relative_path(path)},
                        "trainable": False,
                    },
                }]

    return {
        "info": info,
        "samples": samples,
        "errors": errors,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": matched,
            "has_previous": offset > 0,
            "has_next": offset + len(samples) < matched,
        },
        "query": query,
    }


def normalize_corpus_row(row):
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task_type = row.get("task_type") or metadata.get("task_type") or "chatml"
    system = str(row.get("system") or "")
    instruction = str(row.get("instruction") or row.get("prompt") or "")
    input_text = str(row.get("input") or row.get("source") or row.get("question") or row.get("text") or "")
    output_text = str(row.get("output") or row.get("target") or row.get("answer") or row.get("assistant") or "")

    if "user" in row and "assistant" in row:
        user = str(row.get("user") or "")
        output_text = str(row.get("assistant") or "")
        task_type = metadata.get("task_type") or ("rewrite" if metadata.get("style") or metadata.get("author") else task_type)
        instruction, input_text = split_user_prompt(user, metadata)

    if row.get("messages") and isinstance(row["messages"], list):
        first_user = next((m.get("content", "") for m in row["messages"] if isinstance(m, dict) and m.get("role") == "user"), "")
        last_assistant = next(
            (m.get("content", "") for m in reversed(row["messages"]) if isinstance(m, dict) and m.get("role") == "assistant"),
            "",
        )
        instruction = instruction or "Multi-turn chat sample"
        input_text = str(first_user or "")
        output_text = str(last_assistant or "")

    return {
        "format": "localtune_v1",
        "task_type": task_type,
        "system": system,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": metadata,
        "trainable": bool(input_text and output_text),
    }


def split_user_prompt(user, metadata):
    style = metadata.get("style") or metadata.get("author")
    marker = LOCAL_TUNE_REWRITE_MARKER
    if marker in user:
        before, source = user.split(marker, 1)
        instruction = before.strip()
        if style and "风格" not in instruction:
            instruction = f"改写为指定风格：{style}"
        return instruction or "Rewrite the input in the requested style.", source.strip()

    parts = [part.strip() for part in user.split("\n\n") if part.strip()]
    if len(parts) >= 2:
        return "\n\n".join(parts[:-1]), parts[-1]
    return "Respond to the user input.", user.strip()


def check_corpus_path(path_text, task_type="chatml", dataset_format="chatml_source"):
    path = require_project_path(path_text)
    targets = []
    if path.is_dir():
        targets = sorted(path.rglob("*.jsonl"), key=lambda p: str(p).lower())
    elif path.suffix.lower() == ".jsonl":
        targets = [path]
    else:
        return {
            "ok": False,
            "error": "Format check currently supports JSONL files or folders containing JSONL files.",
            "reports": [],
        }
    reports = [validate_file(target, task_type, dataset_format) for target in targets]
    errors = [msg for report in reports for msg in report.get("errors", [])]
    warnings = [msg for report in reports for msg in report.get("warnings", [])]
    return {
        "ok": not errors,
        "file_count": len(targets),
        "rows": sum(report.get("rows", 0) for report in reports),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "reports": reports,
    }


def derive_corpus_file(payload):
    source = require_project_path(payload.get("source"))
    output = require_project_path(payload.get("output"))
    mode = str(payload.get("mode") or "sample")
    limit = max(1, int(payload.get("limit") or 100))
    seed = int(payload.get("seed") or 42)
    overwrite = bool(payload.get("overwrite", False))
    if output.exists() and not overwrite:
        raise ValueError("Output file already exists. Confirm overwrite before continuing.")

    if source.is_dir():
        inputs = sorted(source.rglob("*.jsonl"), key=lambda p: str(p).lower())
    else:
        if source.suffix.lower() != ".jsonl":
            raise ValueError("Derived datasets can only be created from trainable JSONL files.")
        inputs = [source]
    if not inputs:
        raise ValueError("No JSONL source files found")

    lines = []
    for input_file in inputs:
        if input_file.suffix.lower() != ".jsonl":
            continue
        with input_file.open("r", encoding="utf-8", errors="ignore") as handle:
            lines.extend(line for line in handle if line.strip())
    if not lines:
        raise ValueError("No non-empty JSONL rows found in source files")

    if mode == "sample":
        import random

        rng = random.Random(seed)
        rng.shuffle(lines)
        lines = lines[:limit]
    elif mode == "head":
        lines = lines[:limit]
    elif mode == "tail":
        lines = lines[-limit:]
    elif mode == "copy":
        pass
    else:
        raise ValueError(f"Unknown derive mode: {mode}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
    return {"ok": True, "output": relative_path(output), "rows": len(lines), "size_bytes": output.stat().st_size}


def safe_slug(value, fallback="dataset"):
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or fallback)).strip("_")
    if not text:
        text = fallback
    if not re.match(r"[A-Za-z0-9]", text):
        text = f"{fallback}_{text}"
    return text[:64]


def unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Unable to create a unique file name for {path.name}")


def unique_profile_id(base_id, profiles):
    base_id = normalize_profile_id(safe_slug(base_id, "dataset"))
    if base_id not in profiles:
        return base_id
    for index in range(1, 1000):
        candidate = normalize_profile_id(f"{base_id}_{index}")
        if candidate not in profiles:
            return candidate
    raise ValueError(f"Unable to create a unique dataset profile ID for {base_id}")


@serialized_config_update
def import_corpus_file(payload):
    payload = payload or {}
    selected_path = str(payload.get("path") or "").strip()
    if not selected_path:
        selected_path = choose_local_file(payload.get("initial"))
    if not selected_path:
        return {"cancelled": True}

    source = Path(selected_path).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    if not source.exists() or not source.is_file():
        raise ValueError("Selected corpus file does not exist")
    if source.suffix.lower() != ".jsonl":
        raise ValueError("Training corpus import currently supports JSONL files.")

    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    task_type = str(payload.get("task_type") or data.get("task_type") or "chatml").strip()
    dataset_format = str(payload.get("format") or data.get("dataset_format") or "chatml_source").strip()

    target_dir = PROJECT_ROOT / "data" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)
    if source.parent.resolve() == target_dir.resolve():
        target = source
    else:
        target = unique_path(target_dir / source.name)
        shutil.copy2(source, target)

    profile_id = unique_profile_id(payload.get("profile_id") or target.stem, profiles)
    profile_name = str(payload.get("name") or target.stem).strip()
    profile = {
        "name": profile_name,
        "description": str(payload.get("description") or f"Imported from {source.name}").strip(),
        "task_type": task_type,
        "format": dataset_format,
        "train_file": relative_path(target),
        "val_file": None,
        "test_file": None,
    }
    profiles[profile_id] = profile
    save_project_config(config)

    validation = check_corpus_path(relative_path(target), task_type, dataset_format)
    return {
        "ok": validation.get("ok", False),
        "cancelled": False,
        "profile": profile_id,
        "file": corpus_file_info(target),
        "validation": validation,
        "profiles": get_dataset_profiles(),
    }


@serialized_config_update
def split_dataset_profile(profile_id, val_ratio=0.05, test_ratio=0.05, seed=42):
    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    if profile_id not in profiles:
        raise ValueError(f"Unknown dataset profile: {profile_id}")

    profile = profiles[profile_id] or {}
    train_path = require_project_path(profile.get("train_file"))
    if not train_path.exists() or not train_path.is_file():
        raise ValueError("Training file does not exist")

    val_ratio = max(0.0, min(float(val_ratio), 0.45))
    test_ratio = max(0.0, min(float(test_ratio), 0.45))
    if val_ratio + test_ratio <= 0:
        raise ValueError("At least one of val_ratio or test_ratio must be greater than 0")
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("Validation and test ratios are too large")

    with train_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError("At least 3 training rows are required to split validation/test sets")

    import random
    rng = random.Random(int(seed))
    rng.shuffle(lines)

    n_total = len(lines)
    n_val = max(1 if val_ratio > 0 else 0, int(n_total * val_ratio))
    n_test = max(1 if test_ratio > 0 else 0, int(n_total * test_ratio))
    if n_val + n_test >= n_total:
        n_val = 1 if val_ratio > 0 else 0
        n_test = 1 if test_ratio > 0 and n_total - n_val > 1 else 0

    val_lines = lines[:n_val]
    test_lines = lines[n_val:n_val + n_test]
    train_lines = lines[n_val + n_test:]

    output_dir = train_path.parent
    safe_profile = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in profile_id)
    output_paths = {
        "train_file": output_dir / f"{safe_profile}_split_train.jsonl",
        "val_file": output_dir / f"{safe_profile}_split_val.jsonl",
        "test_file": output_dir / f"{safe_profile}_split_test.jsonl",
    }
    for path, split_lines in [
        (output_paths["train_file"], train_lines),
        (output_paths["val_file"], val_lines),
        (output_paths["test_file"], test_lines),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(split_lines)

    profile["train_file"] = relative_path(output_paths["train_file"])
    profile["val_file"] = relative_path(output_paths["val_file"])
    profile["test_file"] = relative_path(output_paths["test_file"])
    profiles[profile_id] = profile
    save_project_config(config)
    return {
        "ok": True,
        "profile": profile_id,
        "files": {
            "train": profile["train_file"],
            "val": profile["val_file"],
            "test": profile["test_file"],
        },
        "rows": {
            "train": len(train_lines),
            "val": len(val_lines),
            "test": len(test_lines),
        },
    }


def get_dataset_profiles():
    config = load_project_config()
    data = config.get("data", {})
    profiles = data.get("profiles", {}) or {}

    result = []
    for key, profile in profiles.items():
        enriched_profile = dict(profile or {})
        enriched_profile.setdefault("id", key)
        enriched_profile.setdefault("task_type", data.get("task_type", "chatml"))
        enriched_profile.setdefault("format", data.get("dataset_format", "chatml_source"))
        train = file_info(profile.get("train_file"))
        val = file_info(profile.get("val_file"))
        test = file_info(profile.get("test_file"))
        try:
            validation_result = validate_profile(enriched_profile)
            validation = {
                "ok": validation_result.get("ok", False),
                "error_count": validation_result.get("error_count", 0),
                "warning_count": validation_result.get("warning_count", 0),
                "rows": validation_result.get("rows", 0),
            }
        except Exception as exc:
            validation = {"ok": False, "error_count": 1, "warning_count": 0, "message": str(exc)}
        result.append({
            "id": key,
            "name": profile.get("name", key),
            "description": profile.get("description", ""),
            "task_type": enriched_profile.get("task_type"),
            "format": enriched_profile.get("format"),
            "validation": validation,
            "train": train,
            "val": val,
            "test": test,
            "total_rows": sum(v for v in [train.get("rows"), val.get("rows"), test.get("rows")] if v is not None),
            "total_size_bytes": train.get("size_bytes", 0) + val.get("size_bytes", 0) + test.get("size_bytes", 0),
        })
    return result


def normalize_profile_id(value):
    profile_id = str(value or "").strip()
    if not profile_id:
        raise ValueError("语料档案 ID 不能为空")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", profile_id):
        raise ValueError("语料档案 ID 只能包含字母、数字、下划线和短横线，且最长 64 个字符")
    return profile_id


def normalize_profile_file(value, required=False):
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise ValueError("训练集不能为空")
        return None
    path = require_project_path(raw)
    if not path.exists() or not path.is_file():
        raise ValueError(f"语料文件不存在: {raw}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"语料档案只能引用 JSONL 文件: {raw}")
    return relative_path(path)


def normalized_profile_payload(payload):
    return {
        "name": str(payload.get("name") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "task_type": str(payload.get("task_type") or "chatml").strip(),
        "format": str(payload.get("format") or "chatml_source").strip(),
        "train_file": normalize_profile_file(payload.get("train_file"), required=True),
        "val_file": normalize_profile_file(payload.get("val_file")),
        "test_file": normalize_profile_file(payload.get("test_file")),
    }


@serialized_config_update
def create_dataset_profile(payload):
    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    profile_id = normalize_profile_id(payload.get("id"))
    if profile_id in profiles:
        raise ValueError(f"语料档案已存在: {profile_id}")
    profile = normalized_profile_payload(payload)
    profile["name"] = profile["name"] or profile_id
    profiles[profile_id] = profile
    save_project_config(config)
    return {"ok": True, "profile": profile_id, "profiles": get_dataset_profiles()}


@serialized_config_update
def update_dataset_profile(profile_id, payload):
    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    profile_id = normalize_profile_id(profile_id)
    if profile_id not in profiles:
        raise ValueError(f"未知语料档案: {profile_id}")
    new_id = normalize_profile_id(payload.get("id") or profile_id)
    if new_id != profile_id and new_id in profiles:
        raise ValueError(f"语料档案已存在: {new_id}")
    profile = normalized_profile_payload(payload)
    profile["name"] = profile["name"] or new_id
    if new_id == profile_id:
        profiles[profile_id] = profile
    else:
        updated = {}
        for key, value in profiles.items():
            updated[new_id if key == profile_id else key] = profile if key == profile_id else value
        data["profiles"] = updated
    save_project_config(config)
    return {"ok": True, "profile": new_id, "profiles": get_dataset_profiles()}


@serialized_config_update
def copy_dataset_profile(profile_id, payload):
    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    profile_id = normalize_profile_id(profile_id)
    if profile_id not in profiles:
        raise ValueError(f"未知语料档案: {profile_id}")
    target_id = normalize_profile_id(payload.get("id"))
    if target_id in profiles:
        raise ValueError(f"语料档案已存在: {target_id}")
    copied = dict(profiles[profile_id] or {})
    copied["name"] = str(payload.get("name") or f"{copied.get('name') or profile_id} copy").strip()
    copied["description"] = str(payload.get("description") or copied.get("description") or "").strip()
    profiles[target_id] = copied
    save_project_config(config)
    return {"ok": True, "profile": target_id, "profiles": get_dataset_profiles()}


@serialized_config_update
def delete_dataset_profile(profile_id):
    config = load_project_config()
    data = config.setdefault("data", {})
    profiles = data.setdefault("profiles", {})
    profile_id = normalize_profile_id(profile_id)
    if profile_id not in profiles:
        raise ValueError(f"未知语料档案: {profile_id}")
    del profiles[profile_id]
    save_project_config(config)
    return {"ok": True, "profile": profile_id, "profiles": get_dataset_profiles()}


@serialized_config_update
def update_project_config(payload):
    config = load_project_config()
    quant = config.setdefault("quantization", {})
    model = config.setdefault("model", {})
    data = config.setdefault("data", {})
    training = config.setdefault("training", {})
    monitoring = config.setdefault("monitoring", {})
    branches = quant.setdefault("branches", {})

    if "active_branch" in payload:
        branch = str(payload["active_branch"])
        if branch not in branches:
            raise ValueError(f"Unknown quantization branch: {branch}")
        quant["active_branch"] = branch
    if "active_model" in payload:
        model_id = str(payload["active_model"])
        models = model.get("models") or model.get("catalog") or {}
        if models and model_id not in models:
            raise ValueError(f"Unknown model: {model_id}")
        model["active_model"] = model_id
    if "training_output_dir" in payload:
        training["output_dir"] = str(payload["training_output_dir"])
    if "dashboard_port" in payload:
        monitoring["dashboard_port"] = int(payload["dashboard_port"])

    model_paths = payload.get("model_paths") or {}
    for branch, model_path in model_paths.items():
        if branch in branches:
            branches[branch]["model_path"] = str(model_path)

    save_project_config(config)
    return get_project_config_summary()


def get_model_catalog_summary(model_config, branches, accelerator=None):
    catalog = model_config.get("models") or model_config.get("catalog") or {}
    active_model = model_config.get("active_model") or ""
    accelerator = accelerator or {}

    if not catalog:
        return "", []

    if not active_model or active_model not in catalog:
        active_model = next(iter(catalog), "")

    model_items = []
    for model_id, item in catalog.items():
        branch_paths = item.get("paths") or item.get("branch_paths") or item.get("model_paths") or {}
        default_path = item.get("path") or item.get("model_path") or ""
        branch_items = []
        for branch_id, branch in branches.items():
            if not is_supported_training_branch(branch_id, branch):
                continue
            raw_path = branch_paths.get(branch_id) or default_path
            resolved = resolve_project_path(raw_path)
            branch_items.append({
                "id": branch_id,
                "path": raw_path,
                "path_resolved": relative_path(resolved) if resolved else "",
                "path_exists": bool(resolved and resolved.exists()),
                "quant_type": branch.get("quant_type", ""),
                "framework": branch.get("framework", ""),
            })
        model_items.append({
            "id": model_id,
            "name": item.get("name") or model_id,
            "description": item.get("description", ""),
            "active": model_id == active_model,
            "branches": branch_items,
            "suitability": model_hardware_suitability({
                "name": item.get("name") or model_id,
                "path": default_path,
                "paths": branch_paths,
            }, accelerator),
        })

    return active_model, model_items


def is_supported_training_branch(branch_id, branch):
    return not (
        branch_id == "nvfp4"
        or branch.get("load_mode") == "nvfp4_qlora"
        or str(branch.get("quant_type", "")).lower() == "nvfp4"
    )


def infer_model_params_b(model_info):
    text_parts = []
    if isinstance(model_info, dict):
        for key in ("name", "id", "path", "description"):
            text_parts.append(str(model_info.get(key) or ""))
        for value in (model_info.get("paths") or {}).values():
            text_parts.append(str(value or ""))
        config = model_info.get("config") or {}
        if isinstance(config, dict):
            text_parts.extend(str(config.get(key) or "") for key in ("_name_or_path", "model_type"))
    else:
        text_parts.append(str(model_info or ""))

    text = " ".join(text_parts)
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[-_ ]?[Bb](?![A-Za-z])", text)
    if not matches:
        return None
    values = [float(value) for value in matches]
    return max(values)


def _round_gb(value):
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def estimate_qlora_vram_gb(params_b):
    if params_b is None:
        return None
    # Conservative local-console heuristic for NF4/bnb4 QLoRA. It is a
    # guidance signal, not a scheduler guarantee.
    return round(float(params_b) * 0.75 + 2.0, 1)


def _common_model_size_at_or_below(limit_b):
    common_sizes = [3, 7, 8, 14, 27, 32, 35, 70]
    candidates = [size for size in common_sizes if size <= limit_b]
    return max(candidates) if candidates else common_sizes[0]


def model_hardware_suitability(model_info, accelerator=None):
    accelerator = accelerator or {}
    backend = accelerator.get("backend") or "unknown"
    available_vram = _round_gb(accelerator.get("max_memory"))
    params_b = infer_model_params_b(model_info)
    estimated_vram = estimate_qlora_vram_gb(params_b)

    result = {
        "backend": backend,
        "params_b": params_b,
        "estimated_vram_gb": estimated_vram,
        "available_vram_gb": available_vram,
        "load_method": "bnb4",
    }
    if backend != "cuda":
        return {**result, "status": "not_supported", "reason": "cuda_required"}
    if params_b is None:
        return {**result, "status": "unknown", "reason": "unknown_model_size"}
    if not available_vram:
        return {**result, "status": "unknown", "reason": "unknown_vram"}

    if estimated_vram <= available_vram * 0.80:
        return {**result, "status": "recommended", "reason": "fits_with_headroom"}
    if estimated_vram <= available_vram * 0.95:
        return {**result, "status": "caution", "reason": "near_vram_limit"}
    return {**result, "status": "not_recommended", "reason": "likely_exceeds_vram"}


def model_guidance_for_accelerator(accelerator):
    accelerator = accelerator or {}
    backend = accelerator.get("backend") or "unknown"
    vram = _round_gb(accelerator.get("max_memory"))
    guidance = {
        "backend": backend,
        "device_name": accelerator.get("device_name") or "",
        "available_vram_gb": vram,
        "load_method": "bnb4",
        "recommended_max_params_b": None,
        "caution_max_params_b": None,
        "status": "unknown",
    }
    if backend != "cuda":
        return {**guidance, "status": "not_supported", "reason": "cuda_required"}
    if not vram:
        return {**guidance, "status": "unknown", "reason": "unknown_vram"}

    recommended_raw = max(1, int((vram * 0.80 - 2.0) / 0.75))
    caution_raw = max(recommended_raw, int((vram * 0.95 - 2.0) / 0.75))
    recommended = _common_model_size_at_or_below(recommended_raw)
    caution = max(recommended, _common_model_size_at_or_below(caution_raw))
    return {
        **guidance,
        "status": "ready",
        "recommended_max_params_b": recommended,
        "caution_max_params_b": caution,
        "reason": "estimated_from_vram",
    }


def _numeric_version(value):
    parts = re.findall(r"\d+", str(value or "").split("+", 1)[0])
    return tuple(int(part) for part in parts[:3])


def _version_at_least(value, minimum):
    current = _numeric_version(value)
    required = _numeric_version(minimum)
    width = max(len(current), len(required))
    return current + (0,) * (width - len(current)) >= required + (0,) * (width - len(required))


def _dependency_item(item_id, name, version, minimum, required=True, detail=""):
    if not version:
        status = "missing" if required else "optional"
    elif minimum and not _version_at_least(version, minimum):
        status = "incompatible"
    else:
        status = "ready"
    return {
        "id": item_id,
        "name": name,
        "version": version or "",
        "requirement": f">={minimum}" if minimum else "",
        "required": required,
        "status": status,
        "detail": detail,
    }


def _dependency_repair_info(item_id, accelerator):
    backend = (accelerator or {}).get("backend")
    auto_by_launcher = {
        "node": "repairByLauncher",
        "npm": "repairByLauncher",
    }
    auto_by_repair = {
        "torch": "repairTorchCuda" if backend == "cuda" else "repairByLauncher",
        "bitsandbytes": "repairBitsAndBytesCuda" if backend == "cuda" else "repairUnsupportedBackend",
    }
    manual = {
        "python": "repairPythonManual",
        "compute_backend": "repairComputeManual",
        "cuda": "repairCudaManual",
        "nvidia_driver": "repairDriverManual",
        "unsloth": "repairOptionalManual",
    }
    if item_id in auto_by_repair:
        return {"mode": "auto", "hint": auto_by_repair[item_id]}
    if item_id in auto_by_launcher:
        return {"mode": "auto", "hint": auto_by_launcher[item_id]}
    if item_id in manual:
        return {"mode": "manual", "hint": manual[item_id]}
    return {"mode": "launcher", "hint": "repairByLauncher"}


def _package_version(distribution):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception as exc:
        logger.debug("Unable to read package version for %s: %s", distribution, exc, exc_info=True)
        return ""
    if version:
        return str(version)
    try:
        module = importlib.import_module(distribution)
        return str(getattr(module, "__version__", "") or "")
    except Exception as exc:
        logger.debug("Unable to import package %s for version fallback: %s", distribution, exc, exc_info=True)
        return ""


def _torch_accelerator_info():
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "backend": "",
            "device_name": "",
            "device_count": 0,
            "message": str(exc),
            "source": "torch",
        }

    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "available": True,
                "backend": "cuda",
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
                "memory_used": torch.cuda.memory_reserved(0) / 1024**3,
                "max_memory": total,
                "source": "torch-cuda",
            }
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend and mps_backend.is_available():
            return {
                "available": True,
                "backend": "mps",
                "device_name": "Apple Metal / MPS",
                "device_count": 1,
                "source": "torch-mps",
            }
        xpu_backend = getattr(torch, "xpu", None)
        if xpu_backend and xpu_backend.is_available():
            return {
                "available": True,
                "backend": "xpu",
                "device_name": xpu_backend.get_device_name(0),
                "device_count": xpu_backend.device_count(),
                "source": "torch-xpu",
            }
    except Exception as exc:
        return {
            "available": False,
            "backend": "",
            "device_name": "",
            "device_count": 0,
            "message": str(exc),
            "source": "torch",
        }

    return {
        "available": False,
        "backend": "cpu",
        "device_name": "CPU",
        "device_count": 0,
        "message": "No supported accelerator backend available",
        "source": "torch",
    }


def get_environment_dependencies():
    accelerator = _torch_accelerator_info()
    cuda_profile = accelerator.get("backend") == "cuda"
    package_specs = [
        ("torch", "PyTorch", "torch", "2.5.0", True),
        ("transformers", "Transformers", "transformers", "5.8.1", True),
        ("peft", "PEFT", "peft", "0.14.0", True),
        ("trl", "TRL", "trl", "0.9.0", True),
        ("accelerate", "Accelerate", "accelerate", "1.13.0", True),
        ("datasets", "Datasets", "datasets", "3.0.0", True),
        ("bitsandbytes", "bitsandbytes", "bitsandbytes", "0.45.0", cuda_profile),
        ("unsloth", "Unsloth", "unsloth", "2025.5.0", False),
    ]

    python_version = ".".join(str(value) for value in sys.version_info[:3])
    python_ready = sys.version_info >= (3, 12) and sys.version_info < (3, 14)
    items = [{
        "id": "python",
        "name": "Python",
        "version": python_version,
        "requirement": ">=3.12,<3.14",
        "required": True,
        "status": "ready" if python_ready else "incompatible",
        "detail": sys.executable,
    }]

    package_versions = {}
    for item_id, name, distribution, minimum, required in package_specs:
        version = _package_version(distribution)
        package_versions[item_id] = version
        items.append(_dependency_item(item_id, name, version, minimum, required))

    accelerator_version = str(accelerator.get("backend") or "").upper() if accelerator.get("available") else ""
    items.append(_dependency_item(
        "compute_backend",
        "Compute Backend",
        accelerator_version,
        "",
        True,
        accelerator.get("device_name") or accelerator.get("message", ""),
    ))

    torch_cuda_match = re.search(r"\+cu(\d{2,3})", str(package_versions.get("torch") or ""))
    torch_cuda = ""
    if torch_cuda_match:
        digits = torch_cuda_match.group(1)
        torch_cuda = f"{digits[:-1]}.{digits[-1]}"
    cuda_required = accelerator.get("backend") == "cuda"
    items.append(_dependency_item("cuda", "CUDA Runtime", torch_cuda, "12.0", cuda_required, "PyTorch"))

    driver_version = ""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                driver_version = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
        except (OSError, subprocess.SubprocessError):
            pass
    items.append(_dependency_item("nvidia_driver", "NVIDIA Driver", driver_version, "", False, "nvidia-smi"))

    try:
        from scripts.prepare_project import (
            MIN_NODE_MAJOR,
            MIN_NPM_MAJOR,
            command_version,
            frontend_toolchain_candidates,
        )

        candidates = frontend_toolchain_candidates()
        node, npm, source = candidates[0] if candidates else ("", "", "")
        node_version = command_version(node) if node else ""
        npm_version = command_version(npm, extra_path=Path(node).parent) if npm else ""
        items.append(_dependency_item("node", "Node.js", node_version, str(MIN_NODE_MAJOR), True, source))
        items.append(_dependency_item("npm", "npm", npm_version, str(MIN_NPM_MAJOR), True, source))
    except Exception as exc:
        logger.debug("Unable to inspect frontend toolchain: %s", exc, exc_info=True)
        items.append(_dependency_item("node", "Node.js", "", "22", True))
        items.append(_dependency_item("npm", "npm", "", "10", True))

    for item in items:
        item["repair"] = _dependency_repair_info(item["id"], accelerator)

    counts = {
        status: sum(1 for item in items if item["status"] == status)
        for status in ("ready", "missing", "incompatible", "optional")
    }
    counts["required_total"] = sum(1 for item in items if item["required"])
    counts["required_ready"] = sum(
        1 for item in items if item["required"] and item["status"] == "ready"
    )
    return {
        "items": items,
        "counts": counts,
        "accelerator": accelerator,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "checked_at": datetime.now().isoformat(),
    }


def repair_environment_dependencies():
    stdout = io.StringIO()
    stderr = io.StringIO()
    profile = {}
    with redirect_stdout(stdout), redirect_stderr(stderr):
        from scripts.prepare_project import (
            ensure_frontend_dependencies,
            ensure_frontend_toolchain,
            ensure_training_dependency_profile,
        )

        ensure_frontend_toolchain()
        ensure_frontend_dependencies()
        profile = ensure_training_dependency_profile(auto_install=True, write_profile=True)
    return {
        "ok": True,
        "profile": profile,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "dependencies": get_environment_dependencies(),
    }


def normalize_model_dir_value(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value or ""))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_model_scan_dirs(model_config):
    raw_dirs = model_config.get("scan_dirs")
    if raw_dirs is None:
        raw_dirs = model_config.get("model_dirs")
    if not raw_dirs:
        raw_dirs = []

    items = []
    seen = set()
    for raw_value in raw_dirs:
        if not raw_value:
            continue
        try:
            resolved = normalize_model_dir_value(raw_value)
        except Exception:
            resolved = Path(str(raw_value))
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "path": relative_path(resolved),
            "path_resolved": str(resolved),
            "exists": resolved.exists() and resolved.is_dir(),
        })
    return items


def save_model_scan_dirs(config, dir_items):
    model = config.setdefault("model", {})
    model["scan_dirs"] = [item["path"] for item in dir_items]


@serialized_config_update
def update_model_scan_dirs(action, path_value):
    if not path_value:
        raise ValueError("Missing model directory")

    config = load_project_config()
    model = config.setdefault("model", {})
    dirs = get_model_scan_dirs(model)
    target = normalize_model_dir_value(path_value)
    target_key = str(target).lower()

    if action == "add":
        if not target.exists() or not target.is_dir():
            raise ValueError("Model directory does not exist")
        if not any(str(Path(item["path_resolved"])).lower() == target_key for item in dirs):
            dirs.append({
                "path": relative_path(target),
                "path_resolved": str(target),
                "exists": True,
            })
    elif action == "remove":
        dirs = [item for item in dirs if str(Path(item["path_resolved"])).lower() != target_key]
    else:
        raise ValueError("Unknown model directory action")

    save_model_scan_dirs(config, dirs)
    save_project_config(config)
    return get_project_config_summary()


def choose_local_directory(initial_value=None):
    initial = normalize_model_dir_value(initial_value or PROJECT_ROOT)
    if not initial.exists() or not initial.is_dir():
        initial = PROJECT_ROOT.resolve()

    if sys.platform.startswith("win"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select a local model directory'; "
            "$dialog.ShowNewFolderButton = $false; "
            "$dialog.SelectedPath = $env:LOCALTUNE_INITIAL_DIR; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Write-Output $dialog.SelectedPath }"
        )
        env = os.environ.copy()
        env["LOCALTUNE_INITIAL_DIR"] = str(initial)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Sta", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Unable to open the folder picker")
        selected = result.stdout.strip()
        return str(Path(selected).resolve()) if selected else ""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("A native folder picker is not available on this system") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(initialdir=str(initial), mustexist=True) or ""
    finally:
        root.destroy()


def choose_local_file(initial_value=None):
    initial = Path(str(initial_value or PROJECT_ROOT)).expanduser()
    if not initial.is_absolute():
        initial = PROJECT_ROOT / initial
    if initial.is_file():
        initial = initial.parent
    if not initial.exists() or not initial.is_dir():
        initial = PROJECT_ROOT.resolve()

    if sys.platform.startswith("win"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Title = 'Select a JSONL training corpus'; "
            "$dialog.InitialDirectory = $env:LOCALTUNE_INITIAL_DIR; "
            "$dialog.Filter = 'JSONL files (*.jsonl)|*.jsonl|JSON files (*.json)|*.json|All files (*.*)|*.*'; "
            "$dialog.Multiselect = $false; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Write-Output $dialog.FileName }"
        )
        env = os.environ.copy()
        env["LOCALTUNE_INITIAL_DIR"] = str(initial)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Sta", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Unable to open the file picker")
        selected = result.stdout.strip()
        return str(Path(selected).resolve()) if selected else ""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("A native file picker is not available on this system") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            initialdir=str(initial),
            filetypes=[
                ("JSONL files", "*.jsonl"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        ) or ""
    finally:
        root.destroy()


MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin")
GGUF_SUFFIX = ".gguf"
MODEL_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".pytest-tmp",
    ".pytest_cache",
    ".tmp",
    ".uv-cache",
    ".npm-cache",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "tests",
    "outputs",
    "logs",
    "frontend",
    "history-backup",
}


def slugify_model_id(value):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "model")).strip("_").lower()
    if not text:
        text = "model"
    if text[0].isdigit():
        text = f"model_{text}"
    return text[:80]


def read_model_config_json(path):
    try:
        with open(path / "config.json", "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def detect_model_directory(path, accelerator=None):
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    if not config_path.exists():
        return None

    files = [item for item in path.iterdir() if item.is_file()]
    weight_files = [
        item.name for item in files
        if item.name.endswith(MODEL_WEIGHT_SUFFIXES)
        and (
            item.name.startswith(("model", "pytorch_model"))
            or item.name.endswith(".safetensors")
        )
    ]
    if not weight_files:
        return None

    config = read_model_config_json(path)
    quant_cfg = config.get("quantization_config") or {}
    quant_method = str(quant_cfg.get("quant_method") or quant_cfg.get("format") or "").lower()
    # Detect quantization from the model folder itself. Do not inspect the full
    # project-relative path: a repository named "nvfp4-qlora" would otherwise
    # make every local model look like an NVFP4 model.
    path_text = " ".join([path.name, *[str(item) for item in weight_files[:8]]]).lower()
    quant_format = "nvfp4" if "nvfp4" in path_text else (quant_method or "base")
    architectures = config.get("architectures") or []
    model_type = config.get("model_type") or ""
    display_name = config.get("_name_or_path") or path.name
    tokenizer_files = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
    has_tokenizer = any((path / name).exists() for name in tokenizer_files)

    candidate = {
        "id": slugify_model_id(display_name if display_name != "." else path.name),
        "name": display_name if display_name != "." else path.name,
        "description": f"{model_type or 'HF'} · {quant_format}",
        "path": relative_path(path),
        "path_exists": True,
        "model_type": model_type,
        "architectures": architectures,
        "quant_format": quant_format,
        "has_tokenizer": has_tokenizer,
        "weight_files": weight_files[:8],
        "weight_file_count": len(weight_files),
    }
    if accelerator is None:
        accelerator = _cached_golden_check("accelerator", 10, _torch_accelerator_info)
    candidate["suitability"] = model_hardware_suitability({**candidate, "config": config}, accelerator)
    return candidate


def detect_unsupported_model_files(path):
    if not path.is_dir():
        return None
    files = [item for item in path.iterdir() if item.is_file()]
    gguf_files = [item.name for item in files if item.name.lower().endswith(GGUF_SUFFIX)]
    if not gguf_files:
        return None
    return {
        "path": relative_path(path),
        "format": "gguf",
        "file_count": len(gguf_files),
        "files": gguf_files[:8],
        "message": "GGUF is an inference/deployment format and is not supported as a fine-tuning base model.",
    }


def scan_model_directory(root_value, max_dirs=50000, max_candidates=200):
    if not root_value:
        raise ValueError("Missing scan root")
    root = normalize_model_dir_value(root_value)
    if not root.exists() or not root.is_dir():
        raise ValueError("Scan root is not a directory")

    candidates = []
    notices = []
    visited = 0
    accelerator = _cached_golden_check("accelerator", 10, _torch_accelerator_info)
    for current, dirs, _files in os.walk(root):
        visited += 1
        dirs[:] = [name for name in dirs if name not in MODEL_SKIP_DIRS and not name.startswith(".cache")]
        current_path = Path(current)
        candidate = detect_model_directory(current_path, accelerator)
        if candidate:
            candidates.append(candidate)
            dirs[:] = []
            if len(candidates) >= max_candidates:
                break
        else:
            notice = detect_unsupported_model_files(current_path)
            if notice:
                notices.append(notice)
        if visited >= max_dirs:
            break

    return {
        "root": relative_path(root),
        "visited_dirs": visited,
        "truncated": visited >= max_dirs or len(candidates) >= max_candidates,
        "candidates": candidates,
        "notices": notices[:50],
    }


def scan_configured_model_directories():
    config = load_project_config()
    dirs = get_model_scan_dirs(config.get("model", {}) or {})
    scans = []
    candidates = []
    for item in dirs:
        try:
            result = scan_model_directory(item["path_resolved"])
            scans.append({**result, "ok": True})
            candidates.extend(result.get("candidates", []))
        except Exception as exc:
            scans.append({
                "ok": False,
                "root": item.get("path"),
                "error": str(exc),
                "visited_dirs": 0,
                "truncated": False,
                "candidates": [],
            })
    return {
        "scan_dirs": dirs,
        "scans": scans,
        "candidates": candidates,
        "candidate_count": len(candidates),
    }


def _model_provider_for_locale(locale):
    return "modelscope" if str(locale or "").lower().startswith("zh") else "huggingface"


def _model_provider_url(provider, model_name):
    if provider == "modelscope":
        return f"https://modelscope.cn/models/{model_name}"
    return f"https://huggingface.co/{model_name}"


def _model_download_command(provider, model_name, target_dir="models"):
    provider_package = "modelscope" if provider == "modelscope" else "huggingface-hub"
    return (
        f"uv run --isolated --no-project --with {provider_package} "
        f"python scripts/download_model.py --provider {provider} "
        f"--model {model_name} --output {target_dir}"
    )


def _model_download_supported(provider):
    if shutil.which("uv"):
        return provider in {"modelscope", "huggingface", "hf"}
    if provider == "modelscope":
        return importlib.util.find_spec("modelscope") is not None
    if provider in {"huggingface", "hf"}:
        return importlib.util.find_spec("huggingface_hub") is not None
    return False


def _model_download_process_command(provider, model_name, target_dir):
    script_path = str(PROJECT_ROOT / "scripts" / "download_model.py")
    provider_package = "modelscope" if provider == "modelscope" else "huggingface-hub"
    if shutil.which("uv"):
        return [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            provider_package,
            "python",
            script_path,
            "--provider",
            provider,
            "--model",
            model_name,
            "--output",
            str(target_dir),
        ]
    return [
        sys.executable,
        script_path,
        "--provider",
        provider,
        "--model",
        model_name,
        "--output",
        str(target_dir),
    ]


def _model_recommendation_fit(item, accelerator):
    backend = (accelerator or {}).get("backend") or ""
    vram = float((accelerator or {}).get("max_memory") or 0)
    if backend != "cuda":
        return {"status": "unsupported", "reason": "cuda_required"}
    if not vram:
        return {"status": "unknown", "reason": "unknown_vram"}
    if vram >= float(item["recommended_vram_gb"]):
        return {"status": "recommended", "reason": "has_headroom"}
    if vram >= float(item["min_vram_gb"]):
        return {"status": "caution", "reason": "near_limit"}
    return {"status": "too_large", "reason": "insufficient_vram"}


def _download_state_items():
    if not MODEL_DOWNLOAD_STATE_PATH.exists():
        return []
    try:
        data = json.loads(MODEL_DOWNLOAD_STATE_PATH.read_text(encoding="utf-8"))
        return data.get("items", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_download_state_items(items):
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DOWNLOAD_STATE_PATH.write_text(
        json.dumps({"items": items[-50:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _model_download_log_completed(item):
    log_path = resolve_project_path(item.get("log_file"))
    if not log_path or not log_path.exists():
        return False
    try:
        tail = "".join(tail_file(log_path, 40))
    except OSError:
        return False
    return "[YES] Download completed" in tail or "Finish downloading" in tail


def _model_download_completed_path(item):
    log_path = resolve_project_path(item.get("log_file"))
    if not log_path or not log_path.exists():
        return ""
    try:
        for line in reversed(tail_file(log_path, 80)):
            text = line.strip()
            if text.startswith("Model path:"):
                value = text.split(":", 1)[1].strip()
                path = resolve_project_path(value)
                return relative_path(path) if path else value
            if "[YES] Download completed:" in text:
                value = text.split(":", 1)[1].strip()
                path = resolve_project_path(value)
                return relative_path(path) if path else value
    except OSError:
        return ""
    return ""


def _model_download_verified_path(item):
    completed_path = item.get("model_path") or _model_download_completed_path(item)
    path = resolve_project_path(completed_path)
    if not path or not path.exists() or not path.is_dir():
        return ""
    try:
        if detect_model_directory(path):
            return relative_path(path)
    except Exception as exc:
        logger.debug("Unable to verify downloaded model path %s: %s", path, exc, exc_info=True)
    return ""


def _mark_model_download_completed(item, completed_path, now):
    item["status"] = "completed"
    item["returncode"] = 0
    item["finished_at"] = item.get("finished_at") or now
    item["model_path"] = completed_path
    item.pop("error", None)


def _mark_model_download_missing(item, now):
    item["status"] = "missing"
    item["returncode"] = 0
    item["finished_at"] = item.get("finished_at") or now
    item["error"] = "downloaded_model_missing"
    item.pop("model_path", None)


def _refresh_completed_model_download(item, now):
    completed_path = _model_download_verified_path(item)
    if completed_path:
        _mark_model_download_completed(item, completed_path, now)
    else:
        _mark_model_download_missing(item, now)
    return completed_path


def _model_download_processes_for_job(item):
    processes = []
    seen = set()
    root_pid = item.get("pid")
    model_name = str(item.get("model_name") or "")
    target_dir = str(item.get("target_dir") or "")

    def add_process(process):
        if not process or process.pid in seen:
            return
        seen.add(process.pid)
        processes.append(process)

    if root_pid:
        try:
            root = psutil.Process(int(root_pid))
            add_process(root)
            for child in root.children(recursive=True):
                add_process(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass

    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "download_model.py" not in cmdline or model_name not in cmdline:
            continue
        if target_dir and target_dir not in cmdline:
            normalized_target = str(resolve_project_path(target_dir))
            if normalized_target and normalized_target not in cmdline:
                continue
        add_process(process)
    return processes


def _terminate_model_download_processes(item):
    processes = _model_download_processes_for_job(item)
    if not processes:
        return 0

    killed = 0
    # Kill children before parents so uv/python wrappers cannot leave work behind.
    def process_depth(process):
        try:
            return len(process.parents())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    processes.sort(key=process_depth, reverse=True)
    for process in processes:
        try:
            process.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def _refresh_model_download_jobs_locked():
    items = _download_state_items()
    changed = False
    now = datetime.now().isoformat()
    for item in items:
        if item.get("status") not in {"running", "stopping"}:
            if item.get("status") == "completed":
                if not _model_download_verified_path(item):
                    _mark_model_download_missing(item, now)
                    changed = True
                continue
            if item.get("status") == "missing":
                completed_path = _model_download_verified_path(item)
                if completed_path:
                    _mark_model_download_completed(item, completed_path, now)
                    changed = True
                continue
            if item.get("status") == "failed" and _model_download_processes_for_job(item):
                item["status"] = "running"
                item.pop("finished_at", None)
                item.pop("returncode", None)
                changed = True
                logger.info(
                    "Model download still running after parent process exited: job=%s model=%s log=%s",
                    item.get("id"),
                    item.get("model_name"),
                    item.get("log_file"),
                )
            elif item.get("status") == "failed" and _model_download_log_completed(item):
                completed_path = _refresh_completed_model_download(item, now)
                changed = True
                logger.info(
                    "Model download %s according to log: job=%s model=%s log=%s",
                    "completed" if completed_path else "missing",
                    item.get("id"),
                    item.get("model_name"),
                    item.get("log_file"),
                )
            continue
        process = _MODEL_DOWNLOAD_PROCESSES.get(item.get("id"))
        if _model_download_processes_for_job(item):
            continue
        if not process:
            item["status"] = "completed" if _model_download_log_completed(item) else "failed"
            item["finished_at"] = now
            item["returncode"] = 0 if item["status"] == "completed" else item.get("returncode")
            if item["status"] == "failed":
                item["error"] = "process_not_found"
            else:
                _refresh_completed_model_download(item, now)
            changed = True
            logger.info(
                "Model download %s: job=%s model=%s log=%s",
                item["status"],
                item.get("id"),
                item.get("model_name"),
                item.get("log_file"),
            )
            continue
        returncode = process.poll()
        if returncode is None:
            continue
        item["returncode"] = returncode
        item["finished_at"] = now
        if returncode == 0:
            _refresh_completed_model_download(item, now)
        else:
            item["status"] = "failed"
        changed = True
        _MODEL_DOWNLOAD_PROCESSES.pop(item.get("id"), None)
        logger.info(
            "Model download %s: job=%s model=%s returncode=%s log=%s",
            item["status"],
            item.get("id"),
            item.get("model_name"),
            returncode,
            item.get("log_file"),
        )
    if changed:
        _write_download_state_items(items)
    return items


def list_model_download_jobs():
    with _MODEL_DOWNLOAD_LOCK:
        items = _refresh_model_download_jobs_locked()
    for item in items:
        log_path = resolve_project_path(item.get("log_file"))
        item["log_tail"] = "".join(tail_file(log_path, 80)) if log_path and log_path.exists() else ""
    return {"items": sorted(items, key=lambda value: value.get("started_at") or "", reverse=True)}


def get_model_recommendations(locale="en"):
    accelerator = _cached_golden_check("accelerator", 10, _torch_accelerator_info)
    provider = _model_provider_for_locale(locale)
    downloads = list_model_download_jobs()["items"]
    latest_by_model = {}
    for job in downloads:
        latest_by_model.setdefault(job.get("model_id"), job)
    recommendations = []
    for item in DOWNLOADABLE_MODEL_CATALOG:
        fit = _model_recommendation_fit(item, accelerator)
        model_name = item["provider_name"]
        recommendations.append({
            **item,
            "summary": item["summary_zh"] if provider == "modelscope" else item["summary"],
            "fit": fit,
            "provider": provider,
            "provider_model_id": model_name,
            "download_url": _model_provider_url(provider, model_name),
            "download_command": _model_download_command(provider, model_name),
            "download_available": _model_download_supported(provider),
            "download_job": latest_by_model.get(item["id"]),
        })
    priority = {"recommended": 0, "caution": 1, "unknown": 2, "too_large": 3, "unsupported": 4}
    zh_locale = str(locale or "").lower().startswith("zh")

    def language_priority(value):
        family = str(value.get("family") or "").lower()
        if zh_locale:
            return 0 if family == "qwen" else 1
        return 0 if family == "gemma" else 1

    recommendations.sort(key=lambda value: (
        priority.get(value["fit"]["status"], 9),
        language_priority(value),
        value["params_b"],
    ))
    return {
        "ok": True,
        "locale": locale,
        "provider": provider,
        "target_dir": relative_path(PROJECT_ROOT / "models"),
        "accelerator": accelerator,
        "recommendations": recommendations,
        "downloads": downloads,
    }


def start_model_download(payload):
    provider = str(payload.get("provider") or "huggingface").strip().lower()
    model_id = str(payload.get("id") or "").strip()
    model_name = str(payload.get("model") or payload.get("provider_model_id") or "").strip()
    target_dir = resolve_project_path(payload.get("target_dir") or "models")
    if provider == "hf":
        provider = "huggingface"
    if provider not in {"modelscope", "huggingface"}:
        raise ValueError("Unsupported model provider")
    catalog_item = next((item for item in DOWNLOADABLE_MODEL_CATALOG if item["id"] == model_id), None)
    if not catalog_item:
        raise ValueError("Unknown recommended model")
    if not model_name:
        model_name = catalog_item["provider_name"]
    if model_name != catalog_item["provider_name"]:
        raise ValueError("Model id does not match the selected recommendation")
    if not target_dir:
        raise ValueError("Missing target directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    log_file = LOGS_DIR / f"model_download_{job_id}.log"
    command = _model_download_process_command(provider, model_name, target_dir)

    with _MODEL_DOWNLOAD_LOCK:
        items = _refresh_model_download_jobs_locked()
        running_same = next(
            (
                item for item in items
                if item.get("model_id") == model_id and item.get("status") in {"running", "stopping"}
            ),
            None,
        )
        if running_same:
            return {"ok": True, "job": running_same, "already_running": True}
        log_handle = open(log_file, "a", encoding="utf-8", errors="ignore")
        log_handle.write(
            f"[localtune] provider={provider}\n"
            f"[localtune] model={model_name}\n"
            f"[localtune] target_dir={relative_path(target_dir)}\n"
            f"[localtune] command={' '.join(command)}\n\n"
        )
        log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log_handle.close()
        _MODEL_DOWNLOAD_PROCESSES[job_id] = process
        job = {
            "id": job_id,
            "model_id": model_id,
            "model_name": model_name,
            "provider": provider,
            "status": "running",
            "pid": process.pid,
            "target_dir": relative_path(target_dir),
            "log_file": relative_path(log_file),
            "command": [str(part) for part in command],
            "started_at": datetime.now().isoformat(),
        }
        items.append(job)
        _write_download_state_items(items)
        logger.info(
            "Model download started: job=%s model=%s provider=%s pid=%s log=%s target=%s",
            job_id,
            model_name,
            provider,
            process.pid,
            relative_path(log_file),
            relative_path(target_dir),
        )
    return {"ok": True, "job": job}


def cancel_model_download(job_id):
    with _MODEL_DOWNLOAD_LOCK:
        items = _refresh_model_download_jobs_locked()
        process = _MODEL_DOWNLOAD_PROCESSES.get(job_id)
        for item in items:
            if item.get("id") == job_id and item.get("status") in {"running", "stopping", "failed"}:
                killed = _terminate_model_download_processes(item)
                if process and process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                item["status"] = "cancelled" if killed or process else "failed"
                item["finished_at"] = datetime.now().isoformat()
                item["cancelled_at"] = item["finished_at"]
                item["killed_processes"] = killed
                item["returncode"] = -15 if killed else item.get("returncode")
                _MODEL_DOWNLOAD_PROCESSES.pop(job_id, None)
                logger.info(
                    "Model download cancelled: job=%s model=%s killed_processes=%s log=%s",
                    job_id,
                    item.get("model_name"),
                    killed,
                    item.get("log_file"),
                )
        _write_download_state_items(items)
    return {"ok": True, "downloads": list_model_download_jobs()["items"]}


def _export_state_items():
    if not MODEL_EXPORT_STATE_PATH.exists():
        return []
    try:
        data = json.loads(MODEL_EXPORT_STATE_PATH.read_text(encoding="utf-8"))
        return data.get("items", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write_export_state_items(items):
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_EXPORT_STATE_PATH.write_text(
        json.dumps({"items": items[-50:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _processes_for_export_job(item):
    processes = []
    seen = set()
    root_pid = item.get("pid")
    job_id = str(item.get("id") or "")
    adapter_path = str(item.get("adapter_path") or "")

    def add_process(process):
        if not process or process.pid in seen:
            return
        seen.add(process.pid)
        processes.append(process)

    if root_pid:
        try:
            root = psutil.Process(int(root_pid))
            add_process(root)
            for child in root.children(recursive=True):
                add_process(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass

    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "merge_lora.py" not in cmdline:
            continue
        if job_id and job_id not in cmdline and adapter_path and adapter_path not in cmdline:
            continue
        add_process(process)
    return processes


def _terminate_export_processes(item):
    processes = _processes_for_export_job(item)
    if not processes:
        return 0

    def process_depth(process):
        try:
            return len(process.parents())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    processes.sort(key=process_depth, reverse=True)
    terminated = 0
    for process in processes:
        try:
            process.terminate()
            terminated += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return terminated


def _refresh_model_export_jobs_locked():
    items = _export_state_items()
    changed = False
    now = datetime.now().isoformat()
    for item in items:
        if item.get("status") not in {"running", "stopping"}:
            continue
        process = _MODEL_EXPORT_PROCESSES.get(item.get("id"))
        if not process and _processes_for_export_job(item):
            continue
        if not process:
            item["status"] = "failed"
            item["finished_at"] = now
            item["error"] = "process_not_found"
            changed = True
            continue
        returncode = process.poll()
        if returncode is None:
            continue
        item["returncode"] = returncode
        item["status"] = "completed" if returncode == 0 else "failed"
        item["finished_at"] = now
        if returncode == 0:
            item.pop("error", None)
            _write_merged_model_manifest(item)
        else:
            item["error"] = "merge_failed"
        changed = True
        _MODEL_EXPORT_PROCESSES.pop(item.get("id"), None)
        logger.info(
            "Model export %s: job=%s adapter=%s returncode=%s log=%s",
            item["status"],
            item.get("id"),
            item.get("adapter_path"),
            returncode,
            item.get("log_file"),
        )
    if changed:
        _write_export_state_items(items)
    return items


def _adapter_base_model_from_manifest(adapter_path, manifest):
    if not isinstance(manifest, dict):
        return ""
    runtime_config = resolve_project_path((manifest.get("paths") or {}).get("runtime_config"))
    branch = (manifest.get("artifact") or {}).get("branch") or (manifest.get("quantization") or {}).get("active_branch")
    model_id = (manifest.get("run") or {}).get("model_id") or (manifest.get("model") or {}).get("id")
    if runtime_config and runtime_config.exists():
        try:
            config = yaml.safe_load(runtime_config.read_text(encoding="utf-8")) or {}
            model = config.get("model", {}) or {}
            quant = config.get("quantization", {}) or {}
            branch = branch or quant.get("active_branch")
            catalog = model.get("models") or model.get("catalog") or {}
            if model_id and model_id in catalog:
                item = catalog[model_id] or {}
                paths = item.get("paths") or item.get("branch_paths") or item.get("model_paths") or {}
                return paths.get(branch) or item.get("path") or item.get("model_path") or ""
            branch_config = (quant.get("branches") or {}).get(branch or "", {}) or {}
            return branch_config.get("model_path") or ""
        except (OSError, yaml.YAMLError) as exc:
            logger.debug("Unable to read runtime config for export: %s", exc, exc_info=True)
    adapter_check = inspect_adapter(adapter_path)
    return adapter_check.get("base_model") or ""


def _resolve_existing_local_model_path(value):
    if not value:
        return None
    path = resolve_project_path(value)
    if path and path.exists() and path.is_dir():
        return path
    return None


def _base_model_for_adapter(adapter_path, explicit_base_model=""):
    explicit = _resolve_existing_local_model_path(explicit_base_model)
    if explicit:
        return explicit
    manifest = read_artifact_manifest(adapter_path)
    manifest_base = _adapter_base_model_from_manifest(adapter_path, manifest)
    resolved = _resolve_existing_local_model_path(manifest_base)
    if resolved:
        return resolved
    config = load_project_config()
    quant = config.get("quantization", {}) or {}
    branch = (manifest.get("artifact") or {}).get("branch") if isinstance(manifest, dict) else ""
    branch = branch or quant.get("active_branch")
    branch_config = (quant.get("branches") or {}).get(branch or "", {}) or {}
    resolved = _resolve_existing_local_model_path(branch_config.get("model_path"))
    if resolved:
        return resolved
    raise ValueError("Base model directory does not exist. Select or scan the original local model before merging.")


def _default_export_output_path(adapter_path, job_id):
    config = load_project_config()
    training = config.get("training", {}) or {}
    output_root = resolve_project_path(training.get("output_dir", DEFAULT_OUTPUT_DIR)) or OUTPUTS_DIR
    return output_root / "merged" / f"{adapter_path.name}_{job_id}"


def _write_merged_model_manifest(job):
    output_path = resolve_project_path(job.get("output_path"))
    if not output_path or not output_path.exists():
        return None
    manifest = {
        "schema_version": "localtune.merged_model.v1",
        "created_at": datetime.now().isoformat(),
        "artifact": {
            "type": "merged_model",
            "name": output_path.name,
            "path": relative_path(output_path),
        },
        "source": {
            "kind": "lora_merge",
            "job_id": job.get("id"),
            "adapter_path": job.get("adapter_path"),
            "base_model": job.get("base_model"),
            "dtype": job.get("dtype"),
            "log_file": job.get("log_file"),
        },
        "paths": {
            "artifact": relative_path(output_path),
            "log_file": job.get("log_file"),
        },
        "management": {},
    }
    return write_artifact_manifest(output_path, manifest)


def list_model_export_jobs():
    with _MODEL_EXPORT_LOCK:
        items = _refresh_model_export_jobs_locked()
    for item in items:
        log_path = resolve_project_path(item.get("log_file"))
        item["log_tail"] = "".join(tail_file(log_path, 80)) if log_path and log_path.exists() else ""
    return {"items": sorted(items, key=lambda value: value.get("started_at") or "", reverse=True)}


def start_model_export(payload):
    adapter_path = require_artifact_path(payload.get("adapter_path") or payload.get("adapter"))
    adapter_check = inspect_adapter(adapter_path)
    if not adapter_check.get("ok"):
        raise ValueError("Selected artifact is not a valid LoRA adapter")
    base_model = _base_model_for_adapter(adapter_path, payload.get("base_model") or payload.get("base_model_path") or "")
    dtype = str(payload.get("dtype") or "bf16").strip().lower()
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("Unsupported dtype")
    framework = str(payload.get("framework") or "auto").strip().lower()
    if framework not in {"auto", "peft", "unsloth"}:
        raise ValueError("Unsupported merge framework")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_path = resolve_project_path(payload.get("output_path")) if payload.get("output_path") else _default_export_output_path(adapter_path, job_id)
    if not output_path:
        raise ValueError("Missing output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"model_export_{job_id}.log"
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "merge_lora.py"),
        "--lora_path",
        str(adapter_path),
        "--base_model",
        str(base_model),
        "--output_path",
        str(output_path),
        "--dtype",
        dtype,
        "--framework",
        framework,
    ]
    if bool(payload.get("trust_remote_code")):
        command.append("--trust-remote-code")

    with _MODEL_EXPORT_LOCK:
        items = _refresh_model_export_jobs_locked()
        running_same = next(
            (
                item for item in items
                if item.get("adapter_path") == relative_path(adapter_path)
                and item.get("status") in {"running", "stopping"}
            ),
            None,
        )
        if running_same:
            return {"ok": True, "job": running_same, "already_running": True}
        log_handle = open(log_file, "a", encoding="utf-8", errors="ignore")
        log_handle.write(
            "[localtune] action=merge_adapter\n"
            f"[localtune] adapter={relative_path(adapter_path)}\n"
            f"[localtune] base_model={relative_path(base_model)}\n"
            f"[localtune] output_path={relative_path(output_path)}\n"
            f"[localtune] dtype={dtype}\n"
            f"[localtune] command={' '.join(command)}\n\n"
        )
        log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log_handle.close()
        job = {
            "id": job_id,
            "kind": "model_export",
            "status": "running",
            "pid": process.pid,
            "adapter_path": relative_path(adapter_path),
            "base_model": relative_path(base_model),
            "output_path": relative_path(output_path),
            "dtype": dtype,
            "framework": framework,
            "log_file": relative_path(log_file),
            "command": [str(part) for part in command],
            "started_at": datetime.now().isoformat(),
        }
        _MODEL_EXPORT_PROCESSES[job_id] = process
        items.append(job)
        _write_export_state_items(items)
        logger.info(
            "Model export started: job=%s adapter=%s base_model=%s output=%s pid=%s log=%s",
            job_id,
            relative_path(adapter_path),
            relative_path(base_model),
            relative_path(output_path),
            process.pid,
            relative_path(log_file),
        )
    return {"ok": True, "job": job}


def cancel_model_export(job_id):
    with _MODEL_EXPORT_LOCK:
        items = _refresh_model_export_jobs_locked()
        process = _MODEL_EXPORT_PROCESSES.get(job_id)
        for item in items:
            if item.get("id") == job_id and item.get("status") in {"running", "stopping", "failed"}:
                terminated = _terminate_export_processes(item)
                if process and process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                item["status"] = "cancelled" if terminated or process else "failed"
                item["finished_at"] = datetime.now().isoformat()
                item["cancelled_at"] = item["finished_at"]
                item["terminated_processes"] = terminated
                item["returncode"] = -15 if terminated else item.get("returncode")
                _MODEL_EXPORT_PROCESSES.pop(job_id, None)
                logger.info(
                    "Model export cancelled: job=%s adapter=%s terminated_processes=%s log=%s",
                    job_id,
                    item.get("adapter_path"),
                    terminated,
                    item.get("log_file"),
                )
        _write_export_state_items(items)
    return {"ok": True, "exports": list_model_export_jobs()["items"]}


def candidate_branch_paths(candidate, branches):
    quant_format = str(candidate.get("quant_format") or "").lower()
    path = candidate.get("path", "")
    if quant_format == "nvfp4":
        return {}
    return {
        branch_id: path
        for branch_id, branch in branches.items()
        if is_supported_training_branch(branch_id, branch)
    }


def paths_point_to_same_model(left, right):
    if not left or not right:
        return False
    left_path = resolve_project_path(left)
    right_path = resolve_project_path(right)
    try:
        return bool(left_path and right_path and left_path.resolve() == right_path.resolve())
    except OSError:
        return str(left_path) == str(right_path)


@serialized_config_update
def register_model_candidate(payload):
    candidate = payload.get("candidate") or {}
    model_path = candidate.get("path") or payload.get("path")
    path = Path(os.path.expandvars(os.path.expanduser(str(model_path or ""))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    detected = detect_model_directory(path)
    if not detected:
        raise LocalTuneError(
            "MODEL_DIRECTORY_UNRECOGNIZED",
            "Selected path is not a recognized Transformers-compatible base model directory",
        )
    detected.update({key: value for key, value in candidate.items() if value not in (None, "")})

    config = load_project_config()
    quant = config.setdefault("quantization", {})
    branches = quant.setdefault("branches", {})
    model = config.setdefault("model", {})
    catalog = model.setdefault("models", {})
    rel_path = relative_path(path)

    existing_id = None
    for model_id, item in catalog.items():
        paths = item.get("paths") or {}
        if (
            any(paths_point_to_same_model(rel_path, value) for value in paths.values())
            or paths_point_to_same_model(rel_path, item.get("path"))
            or paths_point_to_same_model(rel_path, item.get("model_path"))
        ):
            existing_id = model_id
            break

    model_id = existing_id or slugify_model_id(payload.get("id") or detected.get("id") or detected.get("name"))
    if not existing_id:
        base_id = model_id
        suffix = 2
        while model_id in catalog:
            model_id = f"{base_id}_{suffix}"
            suffix += 1

    paths = payload.get("paths") or candidate_branch_paths({**detected, "path": rel_path}, branches)
    if not paths:
        raise LocalTuneError(
            "MODEL_BRANCH_UNSUPPORTED",
            f"No compatible load method found for this model format: {detected.get('quant_format') or 'unknown'}",
        )

    existing_item = catalog.get(model_id, {}) if existing_id else {}
    existing_paths = existing_item.get("paths") or {}
    merged_paths = {**paths, **existing_paths} if existing_id else paths
    catalog[model_id] = {
        "name": payload.get("name") or existing_item.get("name") or detected.get("name") or model_id,
        "description": payload.get("description") or existing_item.get("description") or detected.get("description", ""),
        "paths": merged_paths,
    }
    if payload.get("make_active", True):
        model["active_model"] = model_id
    save_project_config(config)
    return {
        "ok": True,
        "model_id": model_id,
        "model": catalog[model_id],
        "config": get_project_config_summary(),
    }


@serialized_config_update
def remove_model_catalog_item(model_id):
    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValueError("Missing model id")

    config = load_project_config()
    model = config.setdefault("model", {})
    catalog = model.setdefault("models", {})
    if model_id not in catalog:
        raise LocalTuneError("MODEL_NOT_FOUND", "Model is not in the local model list", 404)

    removed = catalog.pop(model_id)
    if model.get("active_model") == model_id:
        next_active = next(
            (
                item_id for item_id, item in catalog.items()
                if any(
                    resolve_project_path(path_value) and resolve_project_path(path_value).exists()
                    for path_value in (item.get("paths") or {}).values()
                )
            ),
            "",
        )
        model["active_model"] = next_active
    save_project_config(config)
    return {
        "ok": True,
        "removed_model_id": model_id,
        "removed_model": removed,
        "config": get_project_config_summary(),
    }


def get_project_config_summary():
    config = load_project_config()
    quant = config.get("quantization", {}) or {}
    model = config.get("model", {}) or {}
    data = config.get("data", {}) or {}
    training = config.get("training", {}) or {}
    monitoring = config.get("monitoring", {}) or {}
    branches = quant.get("branches", {}) or {}
    runtime_backend = _cached_golden_check("accelerator", 10, _torch_accelerator_info)
    active_model, model_items = get_model_catalog_summary(model, branches, runtime_backend)
    backend_id = runtime_backend.get("backend") or "cpu"

    branch_items = []
    for branch_id, branch in branches.items():
        if not is_supported_training_branch(branch_id, branch):
            continue
        model_path = resolve_project_path(branch.get("model_path"))
        supported_backends = branch.get("supported_backends")
        if not supported_backends:
            supported_backends = ["cuda"] if branch.get("load_mode") in {"bnb_qlora", "unsloth_qlora"} else ["cuda"]
        backend_supported = backend_id in supported_backends
        branch_items.append({
            "id": branch_id,
            "description": branch.get("description", ""),
            "framework": branch.get("framework", ""),
            "load_mode": branch.get("load_mode", ""),
            "quant_type": branch.get("quant_type", ""),
            "supported_backends": supported_backends,
            "compatibility": {
                "backend": backend_id,
                "supported": backend_supported,
                "reason": "" if backend_supported else f"{branch_id} supports {', '.join(supported_backends)}; current backend is {backend_id}",
            },
            "model_path": branch.get("model_path", ""),
            "model_path_resolved": relative_path(model_path) if model_path else "",
            "model_path_exists": bool(model_path and model_path.exists()),
        })

    return {
        "project_root": str(PROJECT_ROOT),
        "config_file": relative_path(BASE_CONFIG_PATH),
        "project": config.get("project", {}),
        "active_model": active_model,
        "models": model_items,
        "model_scan_dirs": get_model_scan_dirs(model),
        "runtime_backend": runtime_backend,
        "active_branch": quant.get("active_branch", ""),
        "branches": branch_items,
        "task_type": data.get("task_type", ""),
        "dataset_format": data.get("dataset_format", ""),
        "training": {
            "output_dir": training.get("output_dir", ""),
            "max_steps": training.get("max_steps"),
            "max_seq_length": training.get("max_seq_length"),
            "learning_rate": training.get("learning_rate"),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        },
        "monitoring": {
            "dashboard_port": monitoring.get("dashboard_port", 6543),
        },
    }


def _golden_step(step_id, label, status, detail, action="", route=""):
    return {
        "id": step_id,
        "label": label,
        "status": status,
        "detail": detail,
        "action": action,
        "route": route,
    }


def _latest_run_with(predicate):
    for record in list_run_records(PROJECT_ROOT):
        try:
            if predicate(record):
                return record
        except Exception:
            continue
    return None


def _golden_training_payload(mode="smoke", model_id="", dataset_profile="", branch="bnb4"):
    config = load_project_config()
    training = config.get("training", {}) or {}
    model = config.get("model", {}) or {}
    lora = model.get("lora", {}) or {}
    smoke = mode == "smoke"
    return {
        "mode": "smoke" if smoke else "formal",
        "branch": branch or "bnb4",
        "model_id": model_id,
        "dataset_profile": dataset_profile,
        "no_fallback": True,
        "max_steps": 1 if smoke else int(training.get("max_steps") or -1),
        "max_seq_length": int(training.get("max_seq_length") or 512),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps") or 16),
        "lora_r": int(lora.get("r") or 16),
        "logging_steps": 1 if smoke else int(training.get("logging_steps") or 10),
        "save_steps": 1 if smoke else int(training.get("save_steps") or 500),
    }


def _golden_training_readiness(
    *,
    training_status,
    environment_ready,
    training_backend_ready,
    model_ready,
    dataset_ready,
):
    if training_status.get("status") in {"running", "stopping"}:
        return {"code": "running", "route": "training", "can_train": False}
    if not training_backend_ready:
        return {"code": "unsupported_backend", "route": "environment", "can_train": False}
    if not environment_ready:
        return {"code": "missing_environment", "route": "environment", "can_train": False}
    if not model_ready:
        return {"code": "missing_model", "route": "models", "can_train": False}
    if not dataset_ready:
        return {"code": "missing_dataset", "route": "corpus", "can_train": False}
    return {"code": "ready", "route": "golden", "can_train": True}


def _training_dependency_counts(dependencies):
    frontend_dependency_ids = {"node", "npm"}
    required_items = [
        item for item in dependencies.get("items", [])
        if item.get("required") and item.get("id") not in frontend_dependency_ids
    ]
    return {
        "required_total": len(required_items),
        "required_ready": sum(1 for item in required_items if item.get("status") == "ready"),
    }


def get_golden_path_status():
    dependencies = _cached_golden_check("dependencies", 10, get_environment_dependencies)
    config = get_project_config_summary()
    profiles = get_dataset_profiles()
    artifacts = list_artifacts()
    gpu = _cached_golden_check("gpu", 5, get_gpu_info)
    training_status = training_manager.status()

    training_dependency_counts = _training_dependency_counts(dependencies)
    required_total = training_dependency_counts["required_total"]
    required_ready = training_dependency_counts["required_ready"]
    accelerator = dependencies.get("accelerator", {}) or {}
    model_guidance = model_guidance_for_accelerator(accelerator)
    runtime_backend = accelerator.get("backend") or "unknown"
    training_backend_ready = runtime_backend == "cuda"
    environment_ready = bool(
        required_total
        and required_ready == required_total
        and gpu.get("available")
        and training_backend_ready
    )

    usable_models = [
        model for model in config.get("models", [])
        if any(branch.get("path_exists") for branch in model.get("branches", []))
    ]
    active_model_id = config.get("active_model") or ""
    selected_model = next((model for model in usable_models if model.get("id") == active_model_id), None)
    selected_model = selected_model or (usable_models[0] if usable_models else None)

    active_branch = config.get("active_branch") or "bnb4"
    selected_branch = None
    if selected_model:
        selected_branch = next(
            (branch for branch in selected_model.get("branches", []) if branch.get("id") == active_branch and branch.get("path_exists")),
            None,
        )
        selected_branch = selected_branch or next(
            (branch for branch in selected_model.get("branches", []) if branch.get("path_exists")),
            None,
        )
    branch_id = (selected_branch or {}).get("id") or active_branch

    ready_profiles = [
        profile for profile in profiles
        if profile.get("validation", {}).get("ok") and profile.get("train", {}).get("exists")
    ]
    selected_profile = ready_profiles[0] if ready_profiles else None

    latest_smoke = _latest_run_with(
        lambda item: item.get("kind", "training") == "training"
        and item.get("mode") == "smoke"
        and item.get("status") == "completed"
    )
    latest_formal = _latest_run_with(
        lambda item: item.get("kind", "training") == "training"
        and item.get("mode") in {"formal", "full"}
        and item.get("status") == "completed"
    )
    latest_evaluation = _latest_run_with(
        lambda item: item.get("kind") == "evaluation" and item.get("status") == "completed"
    )

    adapter_items = [
        item for item in artifacts.get("items", [])
        if item.get("type") == "final_adapter"
        and not item.get("archived")
        and (not item.get("adapter_check") or item.get("adapter_check", {}).get("ok"))
    ]
    best_adapter = next((item for item in adapter_items if item.get("best")), None) or (adapter_items[0] if adapter_items else None)

    model_ready = bool(selected_model and selected_branch)
    dataset_ready = bool(selected_profile)
    plan_ready = environment_ready and model_ready and dataset_ready
    smoke_done = bool(latest_smoke)
    formal_done = bool(latest_formal or best_adapter)
    evaluation_done = bool(latest_evaluation)

    blockers = []
    if not training_backend_ready:
        blockers.append("Training backend is not supported: current release requires NVIDIA CUDA.")
    elif not environment_ready:
        blockers.append("Environment is not ready: check required dependencies and accelerator backend availability.")
    if not model_ready:
        blockers.append("No usable local base model is registered for the active load method.")
    if not dataset_ready:
        blockers.append("No validated Dataset Profile with a training split is available.")

    steps = [
        _golden_step(
            "environment",
            "Environment",
            "done" if environment_ready else "blocked",
            f"{required_ready}/{required_total} required dependencies ready"
            + (f", {gpu.get('device_name')}" if gpu.get("available") else ", accelerator unavailable"),
            "Fix environment",
            "environment",
        ),
        _golden_step(
            "model",
            "Model",
            "done" if model_ready else "blocked",
            selected_model.get("name") if selected_model else "Register a local base model directory",
            "Select model",
            "models",
        ),
        _golden_step(
            "dataset",
            "Dataset",
            "done" if dataset_ready else "blocked",
            f"{selected_profile.get('name')} · {selected_profile.get('validation', {}).get('rows', 0)} rows"
            if selected_profile else "Import a private JSONL corpus and create a Dataset Profile",
            "Import data",
            "corpus",
        ),
        _golden_step(
            "smoke",
            "Test Run",
            "done" if smoke_done else ("ready" if plan_ready else "blocked"),
            f"Last test run: {latest_smoke.get('id')}" if latest_smoke else (
                "Review the test-run recipe and run 1 step to prove model load, data read, loss, and artifact wiring"
            ),
            "Start test run",
            "golden",
        ),
        _golden_step(
            "train",
            "Train",
            "done" if formal_done else ("ready" if smoke_done else "blocked"),
            f"Adapter available: {best_adapter.get('name')}" if best_adapter else "Review formal parameters and start a full training run",
            "Start formal run",
            "training",
        ),
        _golden_step(
            "evaluate",
            "Evaluate",
            "done" if evaluation_done else ("ready" if best_adapter else "blocked"),
            f"Last evaluation: {latest_evaluation.get('id')}" if latest_evaluation else "Compare Base vs Adapter and save a report",
            "Evaluate adapter",
            "inference",
        ),
    ]

    done_count = sum(1 for step in steps if step["status"] == "done")
    ready_bonus = 1 if any(step["status"] == "ready" for step in steps) else 0
    score = round(((done_count + ready_bonus * 0.35) / len(steps)) * 100)
    next_step = next((step for step in steps if step["status"] != "done"), steps[-1])

    smoke_payload = _golden_training_payload(
        "smoke",
        selected_model.get("id") if selected_model else "",
        selected_profile.get("id") if selected_profile else "",
        branch_id,
    )
    formal_payload = _golden_training_payload(
        "formal",
        selected_model.get("id") if selected_model else "",
        selected_profile.get("id") if selected_profile else "",
        branch_id,
    )
    training_readiness = _golden_training_readiness(
        training_status=training_status,
        environment_ready=environment_ready,
        training_backend_ready=training_backend_ready,
        model_ready=model_ready,
        dataset_ready=dataset_ready,
    )

    return {
        "ok": True,
        "checked_at": datetime.now().isoformat(),
        "score": score,
        "steps": steps,
        "next_step": next_step,
        "blockers": blockers,
        "can_start_smoke": plan_ready and training_status.get("status") not in {"running", "stopping"},
        "training_readiness": training_readiness,
        "model_guidance": model_guidance,
        "training_status": training_status,
        "selection": {
            "model": selected_model,
            "branch": selected_branch or {"id": branch_id},
            "dataset_profile": selected_profile,
            "adapter": best_adapter,
        },
        "payloads": {
            "smoke": smoke_payload,
            "formal": formal_payload,
        },
        "metrics": {
            "time_to_evaluated_adapter": "not measured yet",
            "smoke_success": "done" if smoke_done else "pending",
            "data_quality": "ready" if dataset_ready else "needs dataset",
            "evaluation_readiness": "ready" if best_adapter else "waiting for adapter",
        },
    }


def run_project_command(cmd, timeout=300):
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": [str(part) for part in cmd],
    }


def get_inference_base_model(branch=None, model_id=None):
    config = load_project_config()
    quant = config.get("quantization", {})
    model = config.get("model", {}) or {}
    active_branch = branch or quant.get("active_branch", "bnb4")
    branch_config = (quant.get("branches", {}) or {}).get(active_branch, {})

    base_model_raw = branch_config.get("model_path")
    catalog = model.get("models") or model.get("catalog") or {}
    if catalog:
        item = catalog.get(model_id, {}) if model_id else {}
        branch_paths = item.get("paths") or item.get("branch_paths") or item.get("model_paths") or {}
        base_model_raw = branch_paths.get(active_branch) or item.get("path") or item.get("model_path")
    base_model = resolve_project_path(base_model_raw)

    return {
        "branch": active_branch,
        "model_id": model_id or "",
        "base_model": relative_path(base_model) if base_model else "",
        "base_model_exists": bool(base_model and base_model.exists()),
        "available": bool(base_model and base_model.exists()),
    }


def directory_size(path, limit=2000):
    total = 0
    seen = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
                seen += 1
                if seen >= limit:
                    break
    except Exception as exc:
        logger.debug("Failed to count dataset rows in %s: %s", path, exc, exc_info=True)
        return None
    return total


def artifact_info(path, branch, artifact_type, archived=False):
    adapter_file = path / "adapter_model.safetensors"
    manifest = read_artifact_manifest(path)
    manifest_run = manifest.get("run", {}) if isinstance(manifest, dict) else {}
    manifest_paths = manifest.get("paths", {}) if isinstance(manifest, dict) else {}
    management = manifest.get("management", {}) if isinstance(manifest, dict) else {}
    return {
        "name": path.name,
        "branch": branch,
        "type": artifact_type,
        "path": relative_path(path),
        "exists": path.exists(),
        "is_adapter": adapter_file.exists(),
        "size_bytes": directory_size(path) if path.exists() and path.is_dir() else 0,
        "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
        "has_manifest": bool(manifest),
        "manifest": manifest,
        "run_id": manifest_run.get("id"),
        "dataset_profile": manifest_run.get("dataset_profile"),
        "log_file": manifest_paths.get("log_file"),
        "config_file": manifest_paths.get("runtime_config"),
        "archived": bool(archived or management.get("archived_at")),
        "best": bool(management.get("best")),
        "adapter_check": inspect_adapter(path) if artifact_type in {"final_adapter", "checkpoint"} else None,
    }


def list_artifacts():
    config = load_project_config()
    training = config.get("training", {}) or {}
    output_root = resolve_project_path(training.get("output_dir", DEFAULT_OUTPUT_DIR))
    if not output_root or not output_root.exists():
        return {"root": relative_path(output_root) if output_root else "", "items": []}

    items = []
    for branch_dir in sorted([p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith(".")], key=os.path.getmtime, reverse=True):
        branch = branch_dir.name
        final_dir = branch_dir / "final"
        if final_dir.exists():
            items.append(artifact_info(final_dir, branch, "final_adapter"))

        for checkpoint in sorted(branch_dir.glob("checkpoint-*"), key=os.path.getmtime, reverse=True):
            if checkpoint.is_dir():
                items.append(artifact_info(checkpoint, branch, "checkpoint"))

    merged_root = output_root / "merged"
    if merged_root.exists():
        for merged_dir in sorted([p for p in merged_root.iterdir() if p.is_dir()], key=os.path.getmtime, reverse=True):
            items.append(artifact_info(merged_dir, "merged", "merged_model"))

    archive_root = output_root / ".archive"
    if archive_root.exists():
        for branch_dir in sorted([p for p in archive_root.iterdir() if p.is_dir()], key=os.path.getmtime, reverse=True):
            for artifact_dir in sorted([p for p in branch_dir.iterdir() if p.is_dir()], key=os.path.getmtime, reverse=True):
                manifest = read_artifact_manifest(artifact_dir) or {}
                artifact_type = (manifest.get("artifact") or {}).get("type") or "archived"
                if artifact_type == "metrics_run":
                    continue
                items.append(artifact_info(artifact_dir, branch_dir.name, artifact_type, archived=True))

    return {"root": relative_path(output_root), "items": items}


def get_artifact_output_root():
    config = load_project_config()
    training = config.get("training", {}) or {}
    root = resolve_project_path(training.get("output_dir", DEFAULT_OUTPUT_DIR))
    if not root:
        raise ValueError("Artifact output directory is not configured")
    root.mkdir(parents=True, exist_ok=True)
    return root


def require_artifact_path(value):
    path = require_project_path(value)
    root = get_artifact_output_root().resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents or not path.is_dir():
        raise ValueError("Invalid artifact path")
    known_paths = {resolve_project_path(item["path"]).resolve() for item in list_artifacts()["items"]}
    if resolved not in known_paths:
        raise ValueError("Artifact is not managed by LocalTune")
    return path


def update_artifact_management(path, updates):
    manifest = read_artifact_manifest(path)
    if not isinstance(manifest, dict) or manifest.get("error"):
        raise ValueError("Artifact has no valid LocalTune manifest")
    management = manifest.setdefault("management", {})
    management.update(updates)
    write_artifact_manifest(path, manifest)
    return manifest


def manage_artifact(payload):
    action = str(payload.get("action") or "").strip()
    path = require_artifact_path(payload.get("path"))
    root = get_artifact_output_root()
    status = training_manager.status().get("status")
    if action in {"archive", "restore", "delete"} and status in {"running", "stopping"}:
        raise ValueError("Training is running; artifact files cannot be moved or deleted")

    if action == "best":
        selected_manifest = read_artifact_manifest(path)
        if not isinstance(selected_manifest, dict) or selected_manifest.get("error"):
            raise ValueError("Artifact has no valid LocalTune manifest")
        selected_branch = (selected_manifest.get("artifact") or {}).get("branch")
        for manifest_path in root.rglob("localtune_artifact.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (manifest.get("artifact") or {}).get("branch") != selected_branch:
                continue
            manifest.setdefault("management", {})["best"] = manifest_path.parent.resolve() == path.resolve()
            write_artifact_manifest(manifest_path.parent, manifest)
        return {"ok": True, "action": action, "path": relative_path(path)}

    if action == "unbest":
        update_artifact_management(path, {"best": False})
        return {"ok": True, "action": action, "path": relative_path(path)}

    if action == "archive":
        manifest = read_artifact_manifest(path)
        if not isinstance(manifest, dict) or manifest.get("error"):
            raise ValueError("Artifact has no valid LocalTune manifest")
        branch = (manifest.get("artifact") or {}).get("branch") or path.parent.name
        archive_dir = root / ".archive" / branch
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.name}"
        original_path = relative_path(path)
        shutil.move(str(path), str(destination))
        moved_manifest = read_artifact_manifest(destination) or manifest
        moved_manifest.setdefault("management", {}).update({
            "archived_at": datetime.now().isoformat(),
            "original_path": original_path,
            "best": False,
        })
        moved_manifest.setdefault("artifact", {})["path"] = relative_path(destination)
        moved_manifest.setdefault("paths", {})["artifact"] = relative_path(destination)
        write_artifact_manifest(destination, moved_manifest)
        return {"ok": True, "action": action, "path": relative_path(destination)}

    if action == "restore":
        manifest = read_artifact_manifest(path)
        management = manifest.get("management", {}) if isinstance(manifest, dict) else {}
        original_path = resolve_project_path(management.get("original_path"))
        if not original_path:
            raise ValueError("Archived artifact has no original path")
        if original_path.exists():
            raise ValueError("Original artifact path already exists")
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(original_path))
        restored_manifest = read_artifact_manifest(original_path) or manifest
        restored_manifest.setdefault("management", {}).pop("archived_at", None)
        restored_manifest["management"].pop("original_path", None)
        restored_manifest.setdefault("artifact", {})["path"] = relative_path(original_path)
        restored_manifest.setdefault("paths", {})["artifact"] = relative_path(original_path)
        write_artifact_manifest(original_path, restored_manifest)
        return {"ok": True, "action": action, "path": relative_path(original_path)}

    if action == "delete":
        deleted_path = relative_path(path)
        shutil.rmtree(path)
        return {"ok": True, "action": action, "path": deleted_path}

    raise ValueError(f"Unknown artifact action: {action}")


def list_task_logs():
    if not LOGS_DIR.exists():
        return []
    logs = []
    log_paths = list(LOGS_DIR.glob("web_train_*.log")) + list(LOGS_DIR.glob("inference_run_*.log"))
    for path in sorted(log_paths, key=os.path.getmtime, reverse=True):
        status = "unknown"
        is_inference = path.name.startswith("inference_run_")
        job_id = path.stem.replace("web_train_", "").replace("inference_run_", "")
        mode = ""
        branch = ""
        dataset_profile = ""
        tail = tail_file(path, 30)
        for line in tail:
            if is_inference and '"ok": true' in line:
                status = "completed"
            elif is_inference and '"ok": false' in line:
                status = "failed"
            elif "process exited with code 0" in line:
                status = "completed"
            elif "process exited with code" in line:
                status = "failed"
        if is_inference:
            mode = "inference"
        head = head_file(path, 140)
        for line in head:
            if line.startswith("[dashboard] job_id="):
                job_id = line.strip().split("=", 1)[1]
            elif line.startswith("[dashboard] mode="):
                parts = line.strip().replace("[dashboard] ", "").split(",")
                for part in parts:
                    if "=" in part:
                        k, v = part.strip().split("=", 1)
                        if k == "mode":
                            mode = v
                        elif k == "branch":
                            branch = v
            elif line.startswith("[dashboard] dataset_profile="):
                dataset_profile = line.strip().split("=", 1)[1]
        logs.append({
            "file": relative_path(path),
            "name": path.name,
            "job_id": job_id,
            "mode": mode,
            "branch": branch,
            "dataset_profile": dataset_profile,
            "status": status,
            "summary": parse_task_log_summary(head),
            "size_bytes": path.stat().st_size,
            "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        })
    return logs


def parse_task_log_summary(lines):
    summary = {
        "datasets": {},
        "training": {},
        "lora": {},
    }
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("[dashboard] command="):
            summary["command"] = line.split("=", 1)[1]
        elif line.startswith("[dashboard] config="):
            summary["config"] = line.split("=", 1)[1]
        elif line.startswith("[dashboard] quant.active_branch="):
            summary["active_branch"] = line.split("=", 1)[1]
        elif line.startswith("[dashboard] data.active_profile="):
            summary["active_profile"] = line.split("=", 1)[1]
        elif line.startswith("[dashboard] dataset."):
            key, value = line.replace("[dashboard] dataset.", "", 1).split("=", 1)
            summary["datasets"][key] = value
        elif line.startswith("[dashboard] training="):
            summary["training"].update(parse_key_value_text(line.split("=", 1)[1]))
        elif line.startswith("[dashboard] lora="):
            summary["lora"].update(parse_key_value_text(line.split("=", 1)[1]))
        elif line.startswith("[dashboard] output_dir="):
            summary["output_dir"] = line.split("=", 1)[1]
        elif "LoRA r:" in line:
            summary["lora"]["r"] = line.rsplit(":", 1)[-1].strip()
        elif "LoRA alpha:" in line:
            summary["lora"]["alpha"] = line.rsplit(":", 1)[-1].strip()
        elif "最大序列长:" in line:
            summary["training"]["seq"] = line.rsplit(":", 1)[-1].strip()
        elif "学习率:" in line:
            summary["training"]["learning_rate"] = line.rsplit(":", 1)[-1].strip()
        elif "训练轮数:" in line:
            summary["training"]["epochs"] = line.rsplit(":", 1)[-1].strip()
        elif "最大步数:" in line:
            summary["training"]["max_steps"] = line.rsplit(":", 1)[-1].strip()
        elif "批大小:" in line:
            summary["training"]["batch"] = line.rsplit(":", 1)[-1].strip()
        elif "梯度累积:" in line:
            summary["training"]["grad_accum"] = line.rsplit(":", 1)[-1].strip()
        elif "预估VRAM:" in line:
            summary["training"]["estimated_vram"] = line.rsplit(":", 1)[-1].strip()
    return {key: value for key, value in summary.items() if value}


def parse_key_value_text(text):
    result = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        result[key.strip()] = value.strip()
    return result


def tail_file(filepath, n=100):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except Exception as exc:
        logger.debug("Failed to read log tail from %s: %s", filepath, exc, exc_info=True)
        return []


def head_file(filepath, n=30):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _, line in zip(range(n), f):
                lines.append(line)
            return lines
    except Exception as exc:
        logger.debug("Failed to read log head from %s: %s", filepath, exc, exc_info=True)
        return []


def get_gpu_info():
    info = _get_gpu_info_from_nvidia_smi()
    if info:
        return info
    return _torch_accelerator_info()


def get_system_info():
    """Return CPU and memory utilization for the local training host."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "available": True,
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total_gb": memory.total / 1024**3,
            "memory_used_gb": memory.used / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "memory_percent": memory.percent,
            "source": "psutil",
        }
    except Exception as exc:
        logger.debug("psutil system metrics unavailable: %s", exc, exc_info=True)

    if sys.platform.startswith("win"):
        script = (
            "$cpu=(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property LoadPercentage -Average).Average; "
            "$os=Get-CimInstance Win32_OperatingSystem; "
            "[pscustomobject]@{cpu=[math]::Round($cpu,1);"
            "total=[double]$os.TotalVisibleMemorySize;"
            "free=[double]$os.FreePhysicalMemory} | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=3,
                encoding="utf-8",
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                total_kb = float(data.get("total") or 0)
                free_kb = float(data.get("free") or 0)
                used_kb = max(0.0, total_kb - free_kb)
                percent = (used_kb / total_kb * 100) if total_kb else 0.0
                return {
                    "available": True,
                    "cpu_percent": float(data.get("cpu") or 0),
                    "cpu_count": os.cpu_count(),
                    "memory_total_gb": total_kb / 1024**2,
                    "memory_used_gb": used_kb / 1024**2,
                    "memory_available_gb": free_kb / 1024**2,
                    "memory_percent": percent,
                    "source": "powershell",
                }
        except Exception as e:
            return {"available": False, "message": str(e)}

    try:
        load_1m = os.getloadavg()[0]
    except Exception as e:
        return {"available": False, "message": str(e)}
    return {
        "available": True,
        "cpu_percent": None,
        "cpu_count": os.cpu_count(),
        "load_1m": load_1m,
        "source": "os",
    }


def _get_gpu_info_from_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2, encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.debug("nvidia-smi GPU metrics unavailable: %s", exc, exc_info=True)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 5:
        return None
    def optional_float(index):
        if len(parts) <= index:
            return None
        try:
            return float(parts[index])
        except (TypeError, ValueError):
            return None

    used_mb = float(parts[1])
    total_mb = float(parts[2])
    return {
        "available": True,
        "backend": "cuda",
        "device_name": parts[0],
        "device_count": len(result.stdout.strip().splitlines()),
        "memory_used": used_mb / 1024,
        "memory_allocated": used_mb / 1024,
        "memory_reserved": used_mb / 1024,
        "max_memory": total_mb / 1024,
        "gpu_util": float(parts[3]),
        "temperature": float(parts[4]),
        "power_draw_w": optional_float(5),
        "power_limit_w": optional_float(6),
        "source": "nvidia-smi",
    }


def _metric_search_roots(run_id=None):
    if not run_id:
        return [OUTPUTS_DIR]
    safe_run_id = str(run_id).strip()
    if not re.fullmatch(r"[\w.-]+", safe_run_id):
        return []
    roots = []

    for path in OUTPUTS_DIR.rglob(safe_run_id):
        if path.is_dir() and path.parent.name == "runs":
            roots.append(path)

    for manifest_path in OUTPUTS_DIR.rglob("localtune_artifact.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (manifest.get("run") or {}).get("id") != safe_run_id:
            continue
        artifact = manifest.get("artifact") or {}
        if artifact.get("type") != "metrics_run":
            continue
        artifact_path = resolve_project_path((manifest.get("paths") or {}).get("artifact") or artifact.get("path"))
        if artifact_path and artifact_path.is_dir():
            roots.append(artifact_path)

    status = training_manager.status()
    job = status.get("job") or {}
    if job.get("id") == safe_run_id and job.get("started_at"):
        try:
            started_ts = datetime.fromisoformat(job["started_at"]).timestamp() - 5
        except ValueError:
            started_ts = None
        output_dir = resolve_project_path(job.get("output_dir"))
        runs_dir = output_dir / "runs" if output_dir else None
        if started_ts and runs_dir and runs_dir.exists():
            for metrics_file in runs_dir.rglob("metrics.jsonl"):
                try:
                    if metrics_file.stat().st_mtime >= started_ts:
                        roots.append(metrics_file.parent)
                except OSError:
                    continue

    unique_roots = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(root)
    return unique_roots


def get_training_metrics(run_id=None):
    metrics = {
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
        "accuracy": [],
        "latest": {},
        "source": None,
    }
    search_roots = _metric_search_roots(run_id)
    if not search_roots:
        return _add_run_progress(metrics, run_id)
    metrics = _metrics_from_localtune_jsonl(metrics, search_roots)
    if not any(metrics.get(key) for key in ["train_loss", "eval_loss", "learning_rate", "accuracy"]):
        metrics = _metrics_from_trainer_state(metrics, search_roots)
    return _add_run_progress(_add_latest_metrics(metrics), run_id)


def _metrics_from_localtune_jsonl(metrics, search_roots=None):
    metrics_files = []
    for root in search_roots or [OUTPUTS_DIR]:
        metrics_files.extend(root.rglob("metrics.jsonl"))
    if not metrics_files:
        return metrics
    latest_file = max(metrics_files, key=os.path.getmtime)
    try:
        with open(latest_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("metrics") or record
                step = int(record.get("step") or payload.get("step") or 0)
                wall_time = record.get("wall_time")

                def append_metric(series, value):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        item = {"step": step, "value": value}
                        if wall_time is not None:
                            item["wall_time"] = wall_time
                        metrics[series].append(item)

                append_metric("train_loss", payload.get("loss") if "loss" in payload else payload.get("train_loss"))
                append_metric("eval_loss", payload.get("eval_loss"))
                append_metric("learning_rate", payload.get("learning_rate"))
                append_metric("accuracy", payload.get("mean_token_accuracy") if "mean_token_accuracy" in payload else payload.get("accuracy"))
        metrics["source"] = relative_path(latest_file)
    except Exception as exc:
        logger.warning("Failed to read LocalTune metrics from %s: %s", latest_file, exc, exc_info=True)
    return metrics


def _metrics_from_trainer_state(metrics, search_roots=None):
    state_files = []
    for root in search_roots or [OUTPUTS_DIR]:
        state_files.extend(root.rglob("trainer_state.json"))
    if not state_files:
        return metrics
    latest_state = max(state_files, key=os.path.getmtime)
    try:
        with open(latest_state, "r", encoding="utf-8") as f:
            state = json.load(f)
        metrics["trainer_state"] = {
            "global_step": state.get("global_step"),
            "max_steps": state.get("max_steps"),
            "epoch": state.get("epoch"),
        }
        for item in state.get("log_history", []):
            step = item.get("step", 0)
            if "loss" in item:
                metrics["train_loss"].append({"step": step, "value": item["loss"]})
            if "train_loss" in item and "loss" not in item:
                metrics["train_summary_loss"] = {"step": step, "value": item["train_loss"]}
            if "eval_loss" in item:
                metrics["eval_loss"].append({"step": step, "value": item["eval_loss"]})
            if "learning_rate" in item:
                metrics["learning_rate"].append({"step": step, "value": item["learning_rate"]})
        metrics["source"] = str(latest_state.relative_to(PROJECT_ROOT))
    except Exception as exc:
        logger.warning("Failed to read trainer_state metrics from %s: %s", latest_state, exc, exc_info=True)
    return _add_latest_metrics(metrics)


def _add_latest_metrics(metrics):
    latest = {}
    for key in ["train_loss", "eval_loss", "learning_rate", "accuracy"]:
        if metrics.get(key):
            latest[key] = metrics[key][-1]
    metrics["latest"] = latest
    return metrics


def _add_run_progress(metrics, run_id):
    if not run_id:
        metrics["progress"] = None
        return metrics
    record = read_run_record(PROJECT_ROOT, str(run_id)) or {}
    state = metrics.get("trainer_state") or {}
    steps = [
        int(item.get("step") or 0)
        for key in ["train_loss", "eval_loss", "learning_rate", "accuracy"]
        for item in metrics.get(key, [])
    ]
    current_step = max([int(state.get("global_step") or 0), *steps], default=0)
    params = record.get("params") or {}
    total_steps = int(state.get("max_steps") or params.get("max_steps") or 0)
    started_at = record.get("started_at")
    finished_at = record.get("finished_at")
    elapsed_seconds = record.get("elapsed_seconds")
    if not elapsed_seconds and started_at:
        try:
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(finished_at) if finished_at else datetime.now()
            elapsed_seconds = max(0.0, (end - start).total_seconds())
        except (TypeError, ValueError):
            elapsed_seconds = None
    steps_per_second = current_step / elapsed_seconds if current_step and elapsed_seconds else None
    remaining_steps = max(0, total_steps - current_step) if total_steps > 0 else None
    eta_seconds = remaining_steps / steps_per_second if remaining_steps is not None and steps_per_second else None
    metrics["progress"] = {
        "status": record.get("status"),
        "current_step": current_step,
        "total_steps": total_steps,
        "epoch": state.get("epoch"),
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": steps_per_second,
        "eta_seconds": eta_seconds,
        "percent": (current_step / total_steps * 100) if total_steps > 0 else None,
    }
    return metrics


@app.route("/")
def dashboard():
    frontend_dist = get_frontend_dist_dir()
    frontend_index = frontend_dist / "index.html"
    if frontend_index.exists():
        return send_frontend_entry(frontend_dist)
    return render_template("dashboard.html")


@app.route("/assets/<path:filename>")
def dashboard_assets(filename):
    assets_dir = get_frontend_dist_dir() / "assets"
    if assets_dir.exists():
        return send_from_directory(assets_dir, filename)
    return jsonify({"error": "Frontend assets are not built"}), 404


@app.route("/<path:filename>")
def dashboard_static(filename):
    if filename == "api" or filename.startswith("api/"):
        abort(404)
    frontend_dist = get_frontend_dist_dir()
    if frontend_dist.exists() and (frontend_dist / filename).is_file():
        return send_from_directory(frontend_dist, filename)
    return send_frontend_entry(frontend_dist) if (frontend_dist / "index.html").exists() else render_template("dashboard.html")


@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(training_manager.start(payload))
    except Exception as e:
        return api_error_response(e, default_code="TRAINING_START_FAILED")


@app.route("/api/training/stop", methods=["POST"])
def api_training_stop():
    return jsonify(training_manager.stop())


@app.route("/api/training/status")
def api_training_status():
    return jsonify(training_manager.status())


@app.route("/api/golden-path/status")
def api_golden_path_status():
    return jsonify(get_golden_path_status())


@app.route("/api/golden-path/plan", methods=["POST"])
def api_golden_path_plan():
    payload = request.get_json(silent=True) or {}
    status = get_golden_path_status()
    mode = str(payload.get("mode") or "smoke")
    key = "formal" if mode in {"formal", "full"} else "smoke"
    return jsonify({
        "ok": True,
        "mode": key,
        "payload": status.get("payloads", {}).get(key, {}),
        "selection": status.get("selection", {}),
        "blockers": status.get("blockers", []),
        "can_start": status.get("can_start_smoke") if key == "smoke" else status.get("selection", {}).get("adapter") is not None,
    })


@app.route("/api/datasets")
def api_datasets():
    return jsonify({"profiles": get_dataset_profiles()})


@app.route("/api/datasets/profiles", methods=["POST"])
def api_dataset_profiles_create():
    try:
        return jsonify(create_dataset_profile(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/profiles/<profile_id>", methods=["PUT", "DELETE"])
def api_dataset_profile(profile_id):
    try:
        if request.method == "DELETE":
            return jsonify(delete_dataset_profile(profile_id))
        return jsonify(update_dataset_profile(profile_id, request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/profiles/<profile_id>/copy", methods=["POST"])
def api_dataset_profile_copy(profile_id):
    try:
        return jsonify(copy_dataset_profile(profile_id, request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/registry")
def api_datasets_registry():
    return jsonify(build_dataset_registry(PROJECT_ROOT, load_project_config()))


@app.route("/api/models/scan", methods=["POST"])
def api_models_scan():
    try:
        payload = request.get_json(force=True) or {}
        if payload.get("all"):
            return jsonify(scan_configured_model_directories())
        return jsonify(scan_model_directory(payload.get("root")))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/directories", methods=["POST"])
def api_models_directories():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(update_model_scan_dirs(payload.get("action"), payload.get("path")))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/select-directory", methods=["POST"])
def api_models_select_directory():
    try:
        payload = request.get_json(silent=True) or {}
        selected = choose_local_directory(payload.get("initial"))
        return jsonify({"cancelled": not bool(selected), "path": selected})
    except Exception as e:
        return api_error_response(e, status=500)


@app.route("/api/models/register", methods=["POST"])
def api_models_register():
    try:
        return jsonify(register_model_candidate(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/<model_id>", methods=["DELETE"])
def api_models_delete(model_id):
    try:
        return jsonify(remove_model_catalog_item(model_id))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/recommendations")
def api_models_recommendations():
    try:
        return jsonify(get_model_recommendations(request.args.get("locale") or "en"))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/downloads")
def api_models_downloads():
    try:
        return jsonify(list_model_download_jobs())
    except Exception as e:
        return api_error_response(e)


@app.route("/api/models/downloads/start", methods=["POST"])
def api_models_downloads_start():
    try:
        return jsonify(start_model_download(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/models/downloads/<job_id>/cancel", methods=["POST"])
def api_models_downloads_cancel(job_id):
    try:
        return jsonify(cancel_model_download(job_id))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/corpus/library")
def api_corpus_library():
    return jsonify(list_corpus_library())


@app.route("/api/corpus/import", methods=["POST"])
def api_corpus_import():
    try:
        return jsonify(import_corpus_file(request.get_json(silent=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/corpus/preview")
def api_corpus_preview():
    try:
        return jsonify(read_corpus_preview(
            request.args.get("path"),
            request.args.get("limit", 20, type=int),
            request.args.get("offset", 0, type=int),
            request.args.get("query", ""),
        ))
    except Exception as e:
        return api_error_response(e)


@app.route("/api/corpus/check", methods=["POST"])
def api_corpus_check():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(check_corpus_path(
            payload.get("path"),
            payload.get("task_type") or "chatml",
            payload.get("format") or "chatml_source",
        ))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/corpus/derive", methods=["POST"])
def api_corpus_derive():
    try:
        return jsonify(derive_corpus_file(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/validate", methods=["POST"])
def api_datasets_validate():
    payload = request.get_json(force=True) or {}
    profile_id = payload.get("profile")
    profile = profile_from_request(profile_id)
    return jsonify(validate_profile(profile, min_rows=int(payload.get("min_rows") or 1)))


@app.route("/api/datasets/convert", methods=["POST"])
def api_datasets_convert():
    try:
        payload = request.get_json(force=True) or {}
        input_path = require_project_path(payload.get("input"))
        output_path = require_project_path(payload.get("output"))
        task_type = str(payload.get("task_type") or "instruction")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "convert_data.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--task-type",
            task_type,
        ]
        if payload.get("system"):
            cmd.extend(["--system", str(payload["system"])])
        return jsonify(run_project_command(cmd))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/split", methods=["POST"])
def api_datasets_split():
    try:
        payload = request.get_json(force=True) or {}
        input_path = require_project_path(payload.get("input"))
        output_dir = require_project_path(payload.get("output_dir"))
        ratio = str(payload.get("ratio") or "0.9,0.05,0.05")
        seed = int(payload.get("seed") or 42)
        prefix = str(payload.get("prefix") or "")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "split_data.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--ratio",
            ratio,
            "--seed",
            str(seed),
        ]
        if prefix:
            cmd.extend(["--prefix", prefix])
        return jsonify(run_project_command(cmd))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/datasets/profile-split", methods=["POST"])
def api_datasets_profile_split():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(split_dataset_profile(
            str(payload.get("profile") or ""),
            val_ratio=payload.get("val_ratio", 0.05),
            test_ratio=payload.get("test_ratio", 0.05),
            seed=payload.get("seed", 42),
        ))
    except Exception as e:
        return api_error_response(e, include_ok=True)


def profile_from_request(profile_id):
    config = load_project_config()
    data = config.get("data", {}) or {}
    profiles = data.get("profiles", {}) or {}
    selected = str(profile_id or "").strip()
    if not selected:
        raise ValueError("Missing dataset profile")
    if selected not in profiles:
        raise ValueError(f"Unknown dataset profile: {selected}")
    profile = dict(profiles[selected] or {})
    profile.setdefault("id", selected)
    profile.setdefault("task_type", data.get("task_type", "chatml"))
    profile.setdefault("format", data.get("dataset_format", "chatml_source"))
    return profile


@app.route("/api/artifacts")
def api_artifacts():
    return jsonify(list_artifacts())


@app.route("/api/artifacts/manage", methods=["POST"])
def api_artifacts_manage():
    try:
        return jsonify(manage_artifact(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/model-exports")
def api_model_exports():
    return jsonify(list_model_export_jobs())


@app.route("/api/model-exports/start", methods=["POST"])
def api_model_exports_start():
    try:
        return jsonify(start_model_export(request.get_json(force=True) or {}))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/model-exports/<job_id>/cancel", methods=["POST"])
def api_model_exports_cancel(job_id):
    try:
        return jsonify(cancel_model_export(job_id))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        try:
            return jsonify(update_project_config(request.get_json(force=True) or {}))
        except Exception as e:
            return api_error_response(e)
    return jsonify(get_project_config_summary())


@app.route("/api/doctor")
def api_doctor():
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "doctor.py")]
    return jsonify(run_project_command(cmd, timeout=120))


@app.route("/api/environment/dependencies")
def api_environment_dependencies():
    return jsonify(get_environment_dependencies())


@app.route("/api/environment/repair", methods=["POST"])
def api_environment_repair():
    try:
        return jsonify(repair_environment_dependencies())
    except Exception as e:
        return api_error_response(e)


@app.route("/api/logs/history")
def api_logs_history():
    return jsonify({"logs": list_task_logs()})


@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": list_run_records(PROJECT_ROOT)})


@app.route("/api/recipes")
def api_recipes():
    return jsonify({"recipes": list_recipes(PROJECT_ROOT)})


@app.route("/api/recipes/export", methods=["POST"])
def api_recipes_export():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(export_run_recipe(PROJECT_ROOT, str(payload.get("run_id") or ""), payload.get("name")))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/recipes/import", methods=["POST"])
def api_recipes_import():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(import_recipe(PROJECT_ROOT, str(payload.get("path") or "")))
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    payload = request.get_json(force=True) or {}
    requested = payload.get("path")
    target = resolve_project_path(requested)
    if not target:
        return jsonify({"error": "Missing path"}), 400

    try:
        resolved = target.resolve()
        project_root = PROJECT_ROOT.resolve()
        if resolved != project_root and project_root not in resolved.parents:
            return jsonify({"error": "Path is outside the project"}), 403
    except Exception as exc:
        logger.debug("Invalid folder path requested: %s", exc, exc_info=True)
        return jsonify({"error": "Invalid path"}), 400

    if not target.exists():
        return jsonify({"error": "Path does not exist"}), 404

    try:
        if sys.platform.startswith("win"):
            if target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                subprocess.Popen(["explorer", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target.parent if target.is_file() else target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent if target.is_file() else target)])
    except Exception as e:
        return api_error_response(e, status=500)

    return jsonify({"opened": True, "path": relative_path(target)})


@app.route("/api/inference/base-model")
def api_inference_base_model():
    return jsonify(get_inference_base_model(
        request.args.get("branch"),
        request.args.get("model_id"),
    ))


def run_inference_payload(payload, prompt_items=None, run_kind="inference"):
    status = training_manager.status()
    if status.get("status") in {"running", "stopping"}:
        return jsonify({"ok": False, "error": "Training is running. Stop or wait for it before inference."}), 409

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt and not prompt_items:
        return jsonify({"ok": False, "error": "Prompt is required"}), 400

    base_model = resolve_project_path(payload.get("base_model"))
    adapter = resolve_project_path(payload.get("adapter"))
    if not base_model or not base_model.exists():
        return jsonify({"ok": False, "error": "Base model path does not exist"}), 400
    if not adapter or not (adapter / "adapter_model.safetensors").exists():
        return jsonify({"ok": False, "error": "Adapter path does not contain adapter_model.safetensors"}), 400

    max_new_tokens = int(payload.get("max_new_tokens") or 256)
    temperature = float(payload.get("temperature") if payload.get("temperature") is not None else 0.7)
    top_p = float(payload.get("top_p") if payload.get("top_p") is not None else 0.9)
    timeout_seconds = int(payload.get("timeout_seconds") or 1200)
    compare = bool(payload.get("compare"))
    system_prompt = str(payload.get("system_prompt") or "")
    stop_words = payload.get("stop_words") or []
    if isinstance(stop_words, str):
        stop_words = [item.strip() for item in stop_words.split(",") if item.strip()]
    max_input_tokens = max(128, int(payload.get("max_input_tokens") or 4096))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_log = LOGS_DIR / f"inference_run_{run_id}.log"
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "infer_adapter.py"),
        "--base-model",
        str(base_model),
        "--adapter",
        str(adapter),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--system-prompt",
        system_prompt,
        "--stop-words",
        json.dumps(stop_words, ensure_ascii=False),
        "--max-input-tokens",
        str(max_input_tokens),
    ]
    project_config = load_project_config()
    if bool((project_config.get("model") or {}).get("trust_remote_code", False)):
        cmd.append("--trust-remote-code")
    inference_run_dir = run_dir(PROJECT_ROOT, run_id)
    inference_run_dir.mkdir(parents=True, exist_ok=True)
    if prompt_items:
        prompts_file = inference_run_dir / "prompts.json"
        prompts_file.write_text(json.dumps(prompt_items, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd.extend(["--prompts-file", str(prompts_file)])
    else:
        cmd.extend(["--prompt", prompt])
    if compare:
        cmd.append("--compare")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.time()
    create_run_record(PROJECT_ROOT, {
        "id": run_id,
        "kind": run_kind,
        "status": "running",
        "branch": payload.get("branch") or "",
        "started_at": datetime.now().isoformat(),
        "log_file": relative_path(run_log),
        "base_model": relative_path(base_model),
        "adapter": relative_path(adapter),
        "command": [str(part) for part in cmd],
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "prompt": prompt,
            "compare": compare,
            "prompt_count": len(prompt_items or [prompt]),
            "system_prompt": system_prompt,
            "stop_words": stop_words,
            "max_input_tokens": max_input_tokens,
        },
    })
    request_file = inference_run_dir / "request.json"
    request_file.write_text(json.dumps({
        "schema_version": "localtune.inference_request.v1",
        "run_id": run_id,
        "branch": payload.get("branch") or "",
        "base_model": relative_path(base_model),
        "adapter": relative_path(adapter),
        "prompt": prompt,
        "prompts": prompt_items or None,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_seconds": timeout_seconds,
            "compare": compare,
            "system_prompt": system_prompt,
            "stop_words": stop_words,
            "max_input_tokens": max_input_tokens,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    update_run_record(PROJECT_ROOT, run_id, {"request_file": relative_path(request_file)})
    run_log.write_text(
        "[dashboard] mode=inference\n"
        + f"[dashboard] base_model={relative_path(base_model)}\n"
        + f"[dashboard] adapter={relative_path(adapter)}\n"
        + f"[dashboard] max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}\n"
        + f"[dashboard] compare={compare}, prompt_count={len(prompt_items or [prompt])}\n"
        + "[dashboard] prompt=\n"
        + prompt
        + "\n\n[dashboard] command="
        + " ".join(cmd)
        + "\n\n",
        encoding="utf-8",
        errors="ignore",
    )
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        run_log.write_text("Inference timed out after " + str(timeout_seconds) + " seconds\n", encoding="utf-8")
        update_run_record(PROJECT_ROOT, run_id, {
            "status": "failed",
            "finished_at": datetime.now().isoformat(),
            "error": "Inference timed out",
        })
        return jsonify({"ok": False, "error": "Inference timed out", "log_file": relative_path(run_log)}), 504

    with open(run_log, "a", encoding="utf-8", errors="ignore") as f:
        f.write("[stdout]\n" + (result.stdout or "") + "\n\n")
        f.write("[stderr]\n" + (result.stderr or ""))

    parsed = None
    for line in reversed((result.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        parsed = {
            "ok": result.returncode == 0,
            "response": result.stdout.strip() if result.returncode == 0 else "",
            "error": result.stderr.strip() or "Inference process did not return JSON",
        }

    parsed["returncode"] = result.returncode
    parsed["elapsed_seconds"] = parsed.get("elapsed_seconds", time.time() - started)
    parsed["log_file"] = relative_path(run_log)
    parsed["base_model"] = relative_path(base_model)
    parsed["adapter"] = relative_path(adapter)
    parsed["run_id"] = run_id
    result_file = inference_run_dir / "result.json"
    result_file.write_text(json.dumps({
        "schema_version": "localtune.inference_result.v1",
        "run_id": run_id,
        "ok": bool(parsed.get("ok")),
        "response": parsed.get("response") or parsed.get("text") or "",
        "base_response": parsed.get("base_response") or "",
        "adapter_response": parsed.get("adapter_response") or "",
        "results": parsed.get("results") or [],
        "error": parsed.get("error") or "",
        "returncode": result.returncode,
        "elapsed_seconds": parsed["elapsed_seconds"],
        "base_model": relative_path(base_model),
        "adapter": relative_path(adapter),
        "prompt": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "compare": compare,
            "system_prompt": system_prompt,
            "stop_words": stop_words,
            "max_input_tokens": max_input_tokens,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    parsed["result_file"] = relative_path(result_file)
    report_files = {}
    if prompt_items:
        report_json = inference_run_dir / "evaluation_report.json"
        report_md = inference_run_dir / "evaluation_report.md"
        report_payload = {
            "schema_version": "localtune.evaluation_report.v1",
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "base_model": relative_path(base_model),
            "adapter": relative_path(adapter),
            "compare": compare,
            "elapsed_seconds": parsed["elapsed_seconds"],
            "results": parsed.get("results") or [],
        }
        report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = [
            f"# LocalTune Evaluation Report",
            "",
            f"- Run ID: `{run_id}`",
            f"- Base model: `{relative_path(base_model)}`",
            f"- Adapter: `{relative_path(adapter)}`",
            f"- Samples: {len(report_payload['results'])}",
            f"- Elapsed: {parsed['elapsed_seconds']:.1f} s",
            "",
        ]
        for index, item in enumerate(report_payload["results"], 1):
            markdown.extend([
                f"## Sample {index}",
                "",
                "### Prompt",
                "",
                str(item.get("prompt") or ""),
                "",
                "### Expected",
                "",
                str(item.get("expected") or "-"),
                "",
            ])
            if compare:
                markdown.extend([
                    "### Base Model",
                    "",
                    str(item.get("base_response") or "-"),
                    "",
                    "### Adapter",
                    "",
                    str(item.get("adapter_response") or "-"),
                    "",
                ])
            else:
                markdown.extend(["### Response", "", str(item.get("response") or "-"), ""])
        report_md.write_text("\n".join(markdown), encoding="utf-8")
        report_files = {
            "report_json": relative_path(report_json),
            "report_markdown": relative_path(report_md),
        }
        parsed.update(report_files)
    update_run_record(PROJECT_ROOT, run_id, {
        "status": "completed" if result.returncode == 0 and parsed.get("ok") else "failed",
        "returncode": result.returncode,
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": parsed["elapsed_seconds"],
        "response_preview": str(parsed.get("adapter_response") or parsed.get("response") or "")[:1000],
        "error": parsed.get("error", ""),
        "result_file": relative_path(result_file),
        **report_files,
    })
    code = 200 if result.returncode == 0 and parsed.get("ok") else 500
    return jsonify(parsed), code


def evaluation_prompt_from_row(row, index):
    if not isinstance(row, dict):
        return None
    prompt = str(row.get("user") or row.get("prompt") or row.get("instruction") or "")
    input_text = str(row.get("input") or row.get("source") or "")
    if row.get("messages") and isinstance(row["messages"], list):
        prompt = next((str(item.get("content") or "") for item in row["messages"] if isinstance(item, dict) and item.get("role") == "user"), prompt)
    if input_text and input_text not in prompt:
        prompt = f"{prompt}\n\n{input_text}".strip()
    expected = str(row.get("assistant") or row.get("output") or row.get("target") or row.get("answer") or "")
    if not prompt:
        return None
    return {"id": row.get("id") or row.get("qa_id") or f"sample-{index}", "prompt": prompt, "expected": expected}


def load_evaluation_prompts(profile_id, role="test", limit=10):
    profile = profile_from_request(profile_id)
    key = {"train": "train_file", "val": "val_file", "test": "test_file"}.get(role)
    if not key or not profile.get(key):
        raise ValueError(f"Dataset profile has no {role} split")
    path = require_project_path(profile[key])
    prompts = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = evaluation_prompt_from_row(json.loads(line), index)
            except json.JSONDecodeError:
                continue
            if item:
                prompts.append(item)
            if len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError("No usable prompts found in the selected split")
    return prompts


@app.route("/api/inference/run", methods=["POST"])
def api_inference_run():
    return run_inference_payload(request.get_json(force=True) or {})


@app.route("/api/inference/batch", methods=["POST"])
def api_inference_batch():
    try:
        payload = request.get_json(force=True) or {}
        limit = max(1, min(int(payload.get("limit") or 10), 50))
        prompts = load_evaluation_prompts(
            str(payload.get("dataset_profile") or ""),
            role=str(payload.get("role") or "test"),
            limit=limit,
        )
        return run_inference_payload(payload, prompt_items=prompts, run_kind="evaluation")
    except Exception as e:
        return api_error_response(e, include_ok=True)


@app.route("/api/logs")
def api_logs():
    n = request.args.get("n", 300, type=int)
    n = max(50, min(n, 5000))
    requested = request.args.get("file")
    kind = request.args.get("kind")
    if requested:
        log_file = resolve_project_path(requested)
        if not log_file or not log_file.exists() or LOGS_DIR.resolve() not in log_file.resolve().parents:
            return jsonify({"error": "日志文件不存在或不允许访问"}), 404
    else:
        log_file = get_latest_log_file(kind=kind)
    if not log_file:
        return jsonify({"logs": ["No log file"], "file": None})
    lines = tail_file(log_file, n)
    return jsonify({
        "logs": [line.rstrip() for line in lines],
        "file": relative_path(log_file),
        "updated": datetime.now().isoformat(),
    })


@app.route("/api/logs/stream")
def api_logs_stream():
    def generate():
        last_file = None
        last_size = 0
        while True:
            log_file = get_latest_log_file()
            if not log_file:
                yield "data: " + json.dumps({"logs": ["No log file"], "file": None}) + "\n\n"
                time.sleep(2)
                continue
            if log_file != last_file:
                lines = tail_file(log_file, 80)
                yield "data: " + json.dumps({"logs": [l.rstrip() for l in lines], "file": str(log_file.name)}) + "\n\n"
                last_file = log_file
                last_size = os.path.getsize(log_file)
            else:
                current_size = os.path.getsize(log_file)
                if current_size > last_size:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(last_size)
                        lines = f.readlines()
                    yield "data: " + json.dumps({"logs": [l.rstrip() for l in lines], "append": True}) + "\n\n"
                    last_size = current_size
                time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/gpu")
def api_gpu():
    return jsonify(get_gpu_info())


@app.route("/api/system")
def api_system():
    return jsonify(get_system_info())


@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_training_metrics(run_id=request.args.get("job_id") or request.args.get("run_id")))


@app.route("/api/status")
def api_status():
    lf = get_latest_log_file()
    dashboard_port = load_monitoring_config()
    return jsonify({
        "training": training_manager.status(),
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "log_file": str(lf.relative_to(PROJECT_ROOT)) if lf else None,
        "dashboard_port": dashboard_port,
        "timestamp": datetime.now().isoformat(),
    })


def run_dashboard(host=DEFAULT_DASHBOARD_HOST, port=None, debug=False):
    TEMPLATE_DIR.mkdir(exist_ok=True)
    if host == DEFAULT_DASHBOARD_HOST:
        host = os.environ.get("LOCALTUNE_HOST", host)
    if port is None:
        port = load_monitoring_config()
    dashboard_html = TEMPLATE_DIR / "dashboard.html"
    if not dashboard_html.exists():
        print("Warning: template not found: " + str(dashboard_html))
        return

    print("Starting Web Dashboard...")
    print("Access URL: http://" + host + ":" + str(port))
    print("Press Ctrl+C to stop")
    start_golden_path_cache_warmup()
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_dashboard()
