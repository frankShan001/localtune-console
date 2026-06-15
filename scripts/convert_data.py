#!/usr/bin/env python
"""Convert common fine-tuning JSONL formats to chat-source JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SYSTEMS = {
    "instruction": "You are a helpful assistant.",
    "qa": "You answer questions accurately and concisely.",
    "rewrite": "You rewrite the input according to the requested style while preserving meaning.",
    "summary": "You summarize the input faithfully and concisely.",
    "classification": "You classify the input into the requested label set.",
    "code": "You write correct and maintainable code.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def convert_row(row: dict[str, Any], task_type: str, system: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("task_type", task_type)

    if task_type == "instruction":
        instruction = as_text(row.get("instruction") or row.get("user") or row.get("prompt"))
        extra_input = as_text(row.get("input"))
        user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        assistant = as_text(row.get("output") or row.get("assistant") or row.get("completion"))
    elif task_type == "qa":
        user = as_text(row.get("question") or row.get("user"))
        assistant = as_text(row.get("answer") or row.get("assistant"))
    elif task_type == "rewrite":
        style = as_text(row.get("style") or metadata.get("style") or "")
        source = as_text(row.get("source") or row.get("user"))
        target = as_text(row.get("target") or row.get("assistant"))
        user = f"Style: {style}\n\nRewrite the following text:\n\n{source}" if style else f"Rewrite the following text:\n\n{source}"
        assistant = target
        if style:
            metadata.setdefault("style", style)
    elif task_type == "summary":
        user = as_text(row.get("document") or row.get("text") or row.get("user"))
        assistant = as_text(row.get("summary") or row.get("assistant"))
    elif task_type == "classification":
        user = as_text(row.get("text") or row.get("user"))
        assistant = as_text(row.get("label") or row.get("assistant"))
    elif task_type == "code":
        user = as_text(row.get("instruction") or row.get("user") or row.get("prompt"))
        assistant = as_text(row.get("code") or row.get("assistant") or row.get("output"))
    else:
        raise SystemExit(f"convert_data.py does not convert task_type '{task_type}' yet")

    return {
        "system": system,
        "user": user.strip(),
        "assistant": assistant.strip(),
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert common JSONL datasets to chat-source JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["instruction", "qa", "rewrite", "summary", "classification", "code"],
    )
    parser.add_argument("--system", help="Override system prompt.")
    args = parser.parse_args()

    task_type = args.task_type
    system = args.system or DEFAULT_SYSTEMS[task_type]
    rows = [convert_row(row, task_type, system) for row in read_jsonl(Path(args.input))]
    empty = [idx + 1 for idx, row in enumerate(rows) if not row["user"] or not row["assistant"]]
    if empty:
        raise SystemExit(f"Converted rows with empty user/assistant: {empty[:10]}")
    write_jsonl(Path(args.output), rows)
    print(f"Converted {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
