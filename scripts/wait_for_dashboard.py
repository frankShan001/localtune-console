#!/usr/bin/env python
"""Wait for LocalTune Console to become ready before opening the browser."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import webbrowser


def wait_until_ready(health_url: str, timeout: float, interval: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(interval)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open LocalTune Console after its health endpoint is ready."
    )
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--browser-url", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not wait_until_ready(args.health_url, args.timeout, args.interval):
        print(
            f"[WARN] Dashboard did not become ready within {args.timeout:g} seconds. "
            f"Open it manually after checking startup logs: {args.browser_url}",
            flush=True,
        )
        return 1

    print(f"[INFO] Dashboard is ready: {args.browser_url}", flush=True)
    if not args.no_browser:
        webbrowser.open(args.browser_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
