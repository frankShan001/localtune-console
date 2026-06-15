#!/usr/bin/env python
"""Split a JSONL dataset into train/val/test files."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)


def parse_ratio(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratio must contain train,val,test values")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("ratio sum must be positive")
    return parts[0] / total, parts[1] / total, parts[2] / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Split JSONL data into train/val/test files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ratio", type=parse_ratio, default=parse_ratio("0.9,0.05,0.05"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    lines = read_lines(Path(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(lines)

    train_ratio, val_ratio, _ = args.ratio
    n_total = len(lines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": lines[:n_train],
        "val": lines[n_train:n_train + n_val],
        "test": lines[n_train + n_val:],
    }
    output_dir = Path(args.output_dir)
    for name, split_lines in splits.items():
        filename = f"{args.prefix}{name}.jsonl"
        write_lines(output_dir / filename, split_lines)
        print(f"{filename}: {len(split_lines)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
