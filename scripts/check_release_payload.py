#!/usr/bin/env python
"""Reject local runtime data if it is ever added to the Git release payload."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PREFIXES = (
    ".tmp/",
    ".venv/",
    "configs/runtime/",
    "data/",
    "history-backup/",
    "logs/",
    "models/",
    "outputs/",
)
FORBIDDEN_SUFFIXES = (".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors")
ALLOWED_PREFIXES = ("data/.gitkeep", "examples/")


def tracked_files(project_root: Path = PROJECT_ROOT) -> list[str]:
    if not shutil.which("git") or not (project_root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=project_root,
        capture_output=True,
        check=True,
    )
    return [item for item in result.stdout.decode("utf-8").split("\0") if item]


def release_payload_violations(paths: list[str]) -> list[str]:
    violations = []
    for raw_path in paths:
        path = PurePosixPath(raw_path.replace("\\", "/"))
        text = str(path)
        if any(text.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            continue
        if any(text.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(text)
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(text)
    return sorted(set(violations))


def main() -> int:
    paths = tracked_files()
    if not paths:
        print("[skip] Git repository metadata is not available; payload check will run in CI.")
        return 0
    violations = release_payload_violations(paths)
    if violations:
        print("[fail] Local runtime files are tracked:")
        for path in violations:
            print(f"  - {path}")
        return 1
    print("[ok] Git release payload excludes local runtime data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
