#!/usr/bin/env python
"""Prepare local-only configuration and frontend assets for LocalTune Console."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "configs" / "model_config.example.yaml"
RUNTIME_DIR = PROJECT_ROOT / "configs" / "runtime"
DEPENDENCY_PROFILE_PATH = RUNTIME_DIR / "dependency_profile.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist" / "index.html"
LOCAL_NODE_VERSION = "22.14.0"
MIN_NODE_MAJOR = 22
MIN_NPM_MAJOR = 10
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
FRONTEND_SOURCE_PATHS = [
    FRONTEND_DIR / "src",
    FRONTEND_DIR / "index.html",
    FRONTEND_DIR / "package.json",
    FRONTEND_DIR / "package-lock.json",
    FRONTEND_DIR / "vite.config.js",
]


def run_capture(command: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def detect_hardware_backend() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = run_capture(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=10,
        )
        if result and result.returncode == 0:
            first = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
            parts = [part.strip() for part in first.split(",", 1)]
            return {
                "backend": "cuda",
                "label": "NVIDIA CUDA",
                "device_name": parts[0] if parts else "NVIDIA GPU",
                "driver_version": parts[1] if len(parts) > 1 else "",
                "installable": True,
                "requires": ["torch-cuda", "bitsandbytes"],
            }

    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return {
            "backend": "mps",
            "label": "Apple Metal (MPS)",
            "device_name": f"Apple Silicon {platform.machine()}",
            "driver_version": "",
            "installable": False,
            "requires": ["torch"],
        }

    xpu_smi = shutil.which("xpu-smi")
    if xpu_smi:
        return {
            "backend": "xpu",
            "label": "Intel XPU",
            "device_name": "Intel XPU",
            "driver_version": "",
            "installable": False,
            "requires": ["torch-xpu"],
        }

    return {
        "backend": "cpu",
        "label": "CPU",
        "device_name": platform.processor() or "CPU",
        "driver_version": "",
        "installable": False,
        "requires": ["torch"],
    }


def inspect_torch_runtime() -> dict:
    code = r"""
import json
try:
    import torch
    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps and mps.is_available())
    xpu = getattr(torch, "xpu", None)
    xpu_available = bool(xpu and xpu.is_available())
    print(json.dumps({
        "installed": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", "") or "",
        "mps_available": mps_available,
        "xpu_available": xpu_available,
    }))
except Exception as exc:
    print(json.dumps({"installed": False, "error": str(exc)}))
"""
    result = run_capture([sys.executable, "-c", code], timeout=30)
    if not result or result.returncode != 0:
        return {"installed": False, "error": (result.stderr if result else "torch probe failed")}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"installed": False, "error": result.stdout.strip() or result.stderr.strip()}


def module_installed(module: str) -> bool:
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"
    result = run_capture([sys.executable, "-c", code], timeout=15)
    return bool(result and result.returncode == 0)


def uv_pip_install(args: list[str]) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to install hardware-specific training dependencies.")
    run([uv, "pip", "install", "--python", sys.executable, *args], PROJECT_ROOT)


def ensure_training_dependency_profile(auto_install: bool = True, write_profile: bool = True) -> dict:
    profile = detect_hardware_backend()
    torch_info = inspect_torch_runtime()
    installed: list[str] = []
    actions: list[str] = []

    print(f"[setup] Detected compute backend: {profile['label']} ({profile['backend']})")

    if auto_install and profile["backend"] == "cuda":
        torch_ready = bool(torch_info.get("installed") and torch_info.get("cuda_available"))
        if not torch_ready:
            print(f"[setup] Installing CUDA PyTorch from {PYTORCH_CUDA_INDEX} ...")
            uv_pip_install(["--upgrade", "--index-url", PYTORCH_CUDA_INDEX, "torch"])
            installed.append("torch-cuda")
            torch_info = inspect_torch_runtime()
        if not module_installed("bitsandbytes"):
            print("[setup] Installing bitsandbytes for CUDA QLoRA ...")
            uv_pip_install(["--upgrade", "bitsandbytes"])
            installed.append("bitsandbytes")
        if not bool(torch_info.get("cuda_available")):
            actions.append("CUDA hardware was detected, but PyTorch still cannot access CUDA.")
    elif profile["backend"] == "xpu":
        actions.append("Intel XPU detected. Install a compatible PyTorch XPU build manually before training.")
    elif profile["backend"] == "mps":
        actions.append("Apple MPS detected. Current bundled QLoRA branches are CUDA-only; use this state for discovery and dataset work.")
    elif profile["backend"] == "cpu":
        actions.append("No accelerator detected. Training branches that require CUDA will stay disabled until suitable hardware is available.")

    profile.update({
        "python": sys.executable,
        "torch": torch_info,
        "installed": installed,
        "actions": actions,
        "checked_at": int(time.time()),
    })
    if write_profile:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        DEPENDENCY_PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def ensure_local_config() -> bool:
    if CONFIG_PATH.exists():
        return False
    if not CONFIG_EXAMPLE_PATH.exists():
        raise RuntimeError(f"Missing example config: {CONFIG_EXAMPLE_PATH}")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_EXAMPLE_PATH, CONFIG_PATH)
    print(f"[setup] Created local config: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    return True


def newest_source_mtime() -> float:
    mtimes: list[float] = []
    for path in FRONTEND_SOURCE_PATHS:
        if path.is_dir():
            mtimes.extend(item.stat().st_mtime for item in path.rglob("*") if item.is_file())
        elif path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes, default=0)


def frontend_needs_build() -> bool:
    if not FRONTEND_DIST.exists():
        return True
    return newest_source_mtime() > FRONTEND_DIST.stat().st_mtime


def local_executable(name: str) -> Path | None:
    suffix = ".cmd" if sys.platform == "win32" and name == "npm" else ".exe" if sys.platform == "win32" else ""
    candidate = PROJECT_ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / f"{name}{suffix}"
    return candidate if candidate.exists() else None


def find_command(name: str) -> str | None:
    local = local_executable(name)
    if local:
        return str(local)
    return shutil.which(name)


def parse_major_version(value: str) -> int | None:
    text = str(value or "").strip().lstrip("vV")
    first = text.split(".", 1)[0]
    return int(first) if first.isdigit() else None


def command_version(command: str, extra_path: Path | None = None) -> str | None:
    env = os.environ.copy()
    path_entries = [str(Path(command).resolve().parent)]
    if extra_path:
        path_entries.insert(0, str(extra_path.resolve()))
    env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
    try:
        result = subprocess.run(
            [command, "--version"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    return (result.stdout or result.stderr).strip()


def validate_frontend_toolchain(node: str, npm: str) -> tuple[bool, str]:
    node_version = command_version(node)
    npm_version = command_version(npm, extra_path=Path(node).parent)
    node_major = parse_major_version(node_version or "")
    npm_major = parse_major_version(npm_version or "")
    if node_major is None:
        return False, "Node.js version could not be read"
    if node_major < MIN_NODE_MAJOR:
        return False, f"Node.js {node_version} is too old; {MIN_NODE_MAJOR}+ is required"
    if npm_major is None:
        return False, "npm is missing or could not be executed"
    if npm_major < MIN_NPM_MAJOR:
        return False, f"npm {npm_version} is too old; {MIN_NPM_MAJOR}+ is required"
    return True, f"Node.js {node_version}, npm {npm_version}"


def frontend_toolchain_candidates() -> list[tuple[str, str, str]]:
    candidates = []
    local_node = local_executable("node")
    local_npm = local_executable("npm")
    if local_node and local_npm:
        candidates.append((str(local_node), str(local_npm), "project"))
    system_node = shutil.which("node")
    system_npm = shutil.which("npm")
    if system_node and system_npm:
        pair = (str(Path(system_node)), str(Path(system_npm)))
        if not candidates or pair != candidates[0][:2]:
            candidates.append((*pair, "system"))
    return candidates


def run(command: list[str], cwd: Path, extra_path: Path | None = None) -> None:
    print(f"[setup] {' '.join(command)}")
    env = os.environ.copy()
    path_entries = [str(Path(command[0]).resolve().parent)]
    if extra_path:
        path_entries.insert(0, str(extra_path.resolve()))
    env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def ensure_frontend_toolchain() -> tuple[str, str]:
    reasons = []
    for node, npm, source in frontend_toolchain_candidates():
        valid, detail = validate_frontend_toolchain(node, npm)
        if valid:
            print(f"[setup] Using {source} frontend toolchain: {detail}")
            return node, npm
        reasons.append(f"{source}: {detail}")

    venv_dir = PROJECT_ROOT / ".venv"
    if Path(sys.prefix).resolve() != venv_dir.resolve():
        detail = "; ".join(reasons) if reasons else "Node.js/npm were not found"
        raise RuntimeError(
            f"No compatible frontend toolchain: {detail}. "
            "Run `uv sync`, then start LocalTune again."
        )

    if reasons:
        print("[setup] Existing frontend toolchain is incompatible: " + "; ".join(reasons))
    print(f"[setup] Installing Node.js {LOCAL_NODE_VERSION} into .venv...")
    run(
        [
            sys.executable,
            "-m",
            "nodeenv",
            "--node",
            LOCAL_NODE_VERSION,
            "--prebuilt",
            "--python-virtualenv",
            "--force",
        ],
        PROJECT_ROOT,
    )
    node = local_executable("node")
    npm = local_executable("npm")
    if not node or not npm:
        raise RuntimeError("Local Node.js installation completed but node/npm could not be found.")
    valid, detail = validate_frontend_toolchain(str(node), str(npm))
    if not valid:
        raise RuntimeError(f"Local frontend toolchain validation failed: {detail}.")
    print(f"[setup] Local frontend toolchain is ready: {detail}")
    return str(node), str(npm)


def ensure_frontend_dependencies() -> None:
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        return
    node, npm = ensure_frontend_toolchain()
    cache = PROJECT_ROOT / ".npm-cache"
    cache.mkdir(parents=True, exist_ok=True)
    run([npm, "ci", "--cache", str(cache)], FRONTEND_DIR, extra_path=Path(node).parent)


def build_frontend(force: bool = False) -> bool:
    if not force and not frontend_needs_build():
        return False
    ensure_frontend_dependencies()
    node, _ = ensure_frontend_toolchain()
    vite = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"
    if not node or not vite.exists():
        raise RuntimeError("Frontend dependencies are incomplete. Run `npm --prefix frontend ci`.")
    run([node, str(vite), "build", "--configLoader", "native"], FRONTEND_DIR)
    print("[setup] Web console assets are ready.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LocalTune Console for local use.")
    parser.add_argument("--check", action="store_true", help="Check readiness without changing files.")
    parser.add_argument("--force-frontend", action="store_true", help="Rebuild the frontend even when current.")
    parser.add_argument("--skip-frontend", action="store_true", help="Prepare only the local configuration.")
    parser.add_argument(
        "--skip-training-deps",
        action="store_true",
        help="Skip hardware-aware training dependency setup.",
    )
    args = parser.parse_args()

    if args.check:
        issues = []
        if not CONFIG_PATH.exists():
            issues.append("local config is missing")
        if not args.skip_training_deps:
            ensure_training_dependency_profile(auto_install=False, write_profile=False)
        if not args.skip_frontend and frontend_needs_build():
            issues.append("frontend build is missing or stale")
        if issues:
            print("Project preparation required: " + "; ".join(issues))
            return 1
        print("Project preparation check passed.")
        return 0

    ensure_local_config()
    if not args.skip_training_deps:
        ensure_training_dependency_profile(auto_install=True)
    if not args.skip_frontend:
        build_frontend(force=args.force_frontend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
