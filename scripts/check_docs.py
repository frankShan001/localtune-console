#!/usr/bin/env python
"""Check that public documentation has English and Simplified Chinese versions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
]


def chinese_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.zh-CN{path.suffix}")


def public_english_docs() -> list[Path]:
    return [PROJECT_ROOT / path for path in ROOT_DOCS]


def main() -> int:
    missing: list[str] = []
    invalid: list[str] = []
    for english in public_english_docs():
        chinese = chinese_path(english)
        if not english.exists():
            missing.append(str(english.relative_to(PROJECT_ROOT)))
            continue
        if not chinese.exists():
            missing.append(str(chinese.relative_to(PROJECT_ROOT)))
            continue
        english_text = english.read_text(encoding="utf-8")
        chinese_text = chinese.read_text(encoding="utf-8")
        if chinese.name not in english_text:
            invalid.append(f"{english.relative_to(PROJECT_ROOT)}: missing Chinese language link")
        if english.name not in chinese_text:
            invalid.append(f"{chinese.relative_to(PROJECT_ROOT)}: missing English language link")

    if missing or invalid:
        print("Documentation language check failed.")
        for path in missing:
            print(f"[missing] {path}")
        for message in invalid:
            print(f"[invalid] {message}")
        return 1

    print(f"Documentation language check passed: {len(public_english_docs())} pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
