#!/usr/bin/env python
"""Start the LocalTune Console dashboard."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.prepare_project import (  # noqa: E402
    build_frontend,
    ensure_local_config,
    ensure_training_dependency_profile,
)
from src.constants import DEFAULT_DASHBOARD_HOST, DEFAULT_DASHBOARD_PORT  # noqa: E402


def environment_int(name: str, default: int) -> int:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={value!r}")
        return default


def load_monitoring_config() -> int:
    import yaml

    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    monitoring = config.get("monitoring", {})
    return environment_int("LOCALTUNE_PORT", monitoring.get("dashboard_port", DEFAULT_DASHBOARD_PORT))


def start_dashboard(host: str = DEFAULT_DASHBOARD_HOST, port: int = DEFAULT_DASHBOARD_PORT) -> None:
    from src.web_dashboard import run_dashboard

    run_dashboard(host=host, port=port)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=os.environ.get("LOCALTUNE_LOG_LEVEL", "INFO").upper(),
        format="[%(levelname)s] %(message)s",
    )

    ensure_local_config()
    default_port = load_monitoring_config()

    parser = argparse.ArgumentParser(description="Start LocalTune Console")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("LOCALTUNE_HOST", DEFAULT_DASHBOARD_HOST),
        help="Dashboard listen host",
    )
    parser.add_argument("--port", type=int, default=default_port, help="Dashboard port")
    parser.add_argument("--no-frontend-build", action="store_true", help="Skip frontend build check")
    parser.add_argument(
        "--skip-training-deps",
        action="store_true",
        default=os.environ.get("LOCALTUNE_SKIP_TRAINING_DEPS") == "1",
        help="Skip hardware-aware training dependency setup",
    )
    args = parser.parse_args()

    if not args.skip_training_deps:
        ensure_training_dependency_profile(auto_install=True)

    if not args.no_frontend_build:
        build_frontend()

    print("=" * 60)
    print("LocalTune Console")
    print("=" * 60)

    print(f"\nStarting Web Dashboard on port {args.port}...")
    print(f"Access URL: http://localhost:{args.port}")
    print("=" * 60)

    try:
        start_dashboard(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        from src.web_dashboard import training_manager

        training_manager.shutdown()
        print("Services stopped")


if __name__ == "__main__":
    main()
