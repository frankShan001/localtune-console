#!/usr/bin/env python
"""Run cheap release-readiness checks for the local project."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, cmd: list[str], optional: bool = False, cwd: Path = PROJECT_ROOT) -> bool:
    print(f"\n== {name} ==")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=command_env(cmd))
    if result.returncode == 0:
        print(f"[ok] {name}")
        return True
    marker = "warn" if optional else "fail"
    print(f"[{marker}] {name} exited with {result.returncode}")
    return optional


def command_env(command: list[str]) -> dict[str, str] | None:
    executable = Path(command[0])
    if not executable.is_absolute():
        return None
    env = os.environ.copy()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cheap project release checks.")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip local data validation. Useful for clean open-source clones without datasets.",
    )
    parser.add_argument("--skip-frontend", action="store_true", help="Skip the frontend production build.")
    parser.add_argument("--skip-frontend-tests", action="store_true", help="Skip frontend unit tests.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the automated test suite.")
    args = parser.parse_args()

    python = sys.executable
    steps: list[tuple[str, list[str], bool, Path]] = [
        ("python lock", lock_check_command(), False, PROJECT_ROOT),
        ("compile python", [python, "-m", "py_compile", *python_files()], False, PROJECT_ROOT),
        ("documentation languages", [python, "scripts/check_docs.py"], False, PROJECT_ROOT),
        ("release payload", [python, "scripts/check_release_payload.py"], False, PROJECT_ROOT),
        (
            "prepare local config",
            [python, "scripts/prepare_project.py", "--skip-frontend", "--skip-training-deps"],
            False,
            PROJECT_ROOT,
        ),
        ("doctor", [python, "scripts/doctor.py"], True, PROJECT_ROOT),
        ("flask routes", [python, "-m", "flask", "--app", "src.web_dashboard", "routes"], False, PROJECT_ROOT),
    ]
    if not args.skip_tests:
        steps.append(("tests", [python, "-m", "pytest", "-q"], False, PROJECT_ROOT))
    if not args.skip_frontend:
        if not args.skip_frontend_tests:
            steps.append(("frontend tests", frontend_test_command(), False, PROJECT_ROOT / "frontend"))
        steps.append(("frontend build", frontend_build_command(), False, PROJECT_ROOT / "frontend"))
    if not args.skip_data:
        steps.insert(2, ("validate data", [python, "scripts/validate_data.py"], True, PROJECT_ROOT))

    ok = True
    for name, cmd, optional, cwd in steps:
        ok = run_step(name, cmd, optional=optional, cwd=cwd) and ok

    print("\nRelease check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def python_files() -> list[str]:
    files = []
    for folder in ("src", "scripts"):
        files.extend(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / folder).rglob("*.py"))
    return files


def frontend_build_command() -> list[str]:
    npm = local_or_system_npm()
    if npm:
        return [npm, "run", "build"]
    local_node = PROJECT_ROOT / ".venv" / "Scripts" / "node.exe"
    vite = PROJECT_ROOT / "frontend" / "node_modules" / "vite" / "bin" / "vite.js"
    if local_node.exists() and vite.exists():
        return [str(local_node), str(vite), "build", "--configLoader", "native"]
    return [sys.executable, "-c", "raise SystemExit('npm is required for the frontend build')"]


def frontend_test_command() -> list[str]:
    npm = local_or_system_npm()
    if not npm:
        return [sys.executable, "-c", "raise SystemExit('npm is required for frontend tests')"]
    return [npm, "run", "test"]


def local_or_system_npm() -> str | None:
    local_npm = PROJECT_ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "npm.cmd" if sys.platform == "win32" else "npm"
    )
    if local_npm.exists():
        return str(local_npm)
    return shutil.which("npm")


def lock_check_command() -> list[str]:
    uv = shutil.which("uv")
    if not uv:
        return [sys.executable, "-c", "raise SystemExit('uv is required to verify uv.lock')"]
    return [uv, "lock", "--check"]


if __name__ == "__main__":
    sys.exit(main())
