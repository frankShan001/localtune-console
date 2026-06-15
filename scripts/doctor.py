#!/usr/bin/env python
"""Lightweight environment and project layout check."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("LocalTune Console doctor")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {PROJECT_ROOT}")

    checks = []

    for path in [
        CONFIG_PATH,
        PROJECT_ROOT / "scripts" / "train.py",
        PROJECT_ROOT / "src" / "web_dashboard.py",
        PROJECT_ROOT / "templates" / "dashboard.html",
    ]:
        ok = path.exists()
        checks.append(ok)
        print(f"{'OK' if ok else 'MISSING'}: {path.relative_to(PROJECT_ROOT)}")

    print("\nPython modules")
    required = ["yaml", "flask", "psutil"]
    optional = ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "unsloth"]
    for name in required:
        ok = check_module(name)
        checks.append(ok)
        print(f"{'OK' if ok else 'MISSING'}: {name}")
    for name in optional:
        status = "OK" if check_module(name) else "optional missing"
        print(f"{status}: {name}")

    if check_module("torch"):
        import torch

        print("\nAccelerator backend")
        cuda_backend = getattr(torch, "cuda", None)
        if cuda_backend and cuda_backend.is_available():
            props = cuda_backend.get_device_properties(0)
            print("backend: CUDA")
            print(f"device: {props.name}")
            print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"CUDA: {torch.version.cuda}")
            print(f"BF16: {cuda_backend.is_bf16_supported()}")
        elif getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
            print("backend: Apple Metal / MPS")
            print("device: Apple Silicon GPU")
        elif getattr(torch, "xpu", None) and torch.xpu.is_available():
            print("backend: Intel XPU")
            print(f"device: {torch.xpu.get_device_name(0)}")
        else:
            print("backend: unavailable")

    print("\nConfig")
    if CONFIG_PATH.exists() and check_module("yaml"):
        import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        branch = config.get("quantization", {}).get("active_branch")
        branches = config.get("quantization", {}).get("branches", {})
        model_path = branches.get(branch, {}).get("model_path")
        print(f"active_branch: {branch}")
        print(f"model_path: {model_path}")
        if model_path and model_path.startswith(("./", ".\\")):
            resolved = (PROJECT_ROOT / model_path).resolve()
            print(f"resolved_model_path: {resolved}")
            print(f"model_path_exists: {resolved.exists()}")

    ok = all(checks)
    print("\nResult:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
