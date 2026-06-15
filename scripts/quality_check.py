#!/usr/bin/env python
"""Run the local quality gate used before release or larger UI changes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def command_env(command: list[str]) -> dict[str, str] | None:
    executable = Path(command[0])
    if not executable.is_absolute():
        return None
    env = os.environ.copy()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    return env


def run_step(name: str, command: list[str], cwd: Path = PROJECT_ROOT) -> bool:
    print(f"\n== {name} ==")
    print(" ".join(command))
    result = subprocess.run(command, cwd=cwd, env=command_env(command))
    if result.returncode == 0:
        print(f"[ok] {name}")
        return True
    print(f"[fail] {name} exited with {result.returncode}")
    return False


def npm_command() -> str:
    local_npm = PROJECT_ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "npm.cmd" if sys.platform == "win32" else "npm"
    )
    if local_npm.exists():
        return str(local_npm)
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required for frontend tests. Run LocalTune setup or install Node.js 22+.")
    return npm


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LocalTune quality checks.")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend tests and build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip frontend production build.")
    parser.add_argument("--skip-release-check", action="store_true", help="Skip release-readiness checks.")
    args = parser.parse_args()

    steps: list[tuple[str, list[str], Path]] = [
        ("python tests", [sys.executable, "-m", "pytest", "-q"], PROJECT_ROOT),
    ]

    if not args.skip_frontend:
        npm = npm_command()
        steps.append(("frontend tests", [npm, "run", "test"], FRONTEND_DIR))
        if not args.skip_build:
            steps.append(("frontend build", [npm, "run", "build"], FRONTEND_DIR))

    if not args.skip_release_check:
        steps.append((
            "release check",
            [sys.executable, "scripts/release_check.py", "--skip-tests", "--skip-frontend", "--skip-data"],
            PROJECT_ROOT,
        ))

    ok = True
    for name, command, cwd in steps:
        ok = run_step(name, command, cwd=cwd) and ok

    print("\nQuality check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
