#!/usr/bin/env python
"""Validate local LLM fine-tuning datasets.

The validator is task-aware but content-neutral. It checks whether data is
usable for common fine-tuning scenarios without enforcing project-specific
repair logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a project dependency.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REWRITE_MARKER = "\u8bf7\u628a\u4e0b\u9762\u6587\u7ae0\u6539\u5199\u6210\u4e0a\u8ff0\u98ce\u683c\uff1a"
OLD_TASK_MARKERS = ("\u7eed\u5199", "continuation", "style_imitation")

TASK_SCHEMAS: dict[str, dict[str, Any]] = {
    "chatml": {
        "formats": ["chatml_source"],
        "required": ["system", "user", "assistant"],
        "input": "user",
        "output": "assistant",
    },
    "instruction": {
        "formats": ["alpaca", "chatml_source"],
        "required_any": [["instruction", "output"], ["user", "assistant"]],
        "input": "instruction",
        "output": "output",
    },
    "chat": {
        "formats": ["messages"],
        "required": ["messages"],
    },
    "qa": {
        "formats": ["qa", "chatml_source"],
        "required_any": [["question", "answer"], ["user", "assistant"]],
        "input": "question",
        "output": "answer",
    },
    "rewrite": {
        "formats": ["source_target", "chatml_source"],
        "required_any": [["source", "target"], ["user", "assistant"]],
        "input": "source",
        "output": "target",
        "length_ratio_warning": 8,
    },
    "summary": {
        "formats": ["document_summary", "chatml_source"],
        "required_any": [["document", "summary"], ["user", "assistant"]],
        "input": "document",
        "output": "summary",
    },
    "classification": {
        "formats": ["text_label", "chatml_source"],
        "required_any": [["text", "label"], ["user", "assistant"]],
        "input": "text",
        "output": "label",
    },
    "extraction": {
        "formats": ["text_json", "chatml_source"],
        "required_any": [["text", "output"], ["user", "assistant"]],
        "input": "text",
        "output": "output",
        "json_output": True,
    },
    "tool_calling": {
        "formats": ["messages"],
        "required": ["messages"],
        "allow_tool_calls": True,
    },
    "code": {
        "formats": ["instruction_code", "chatml_source"],
        "required_any": [["instruction", "code"], ["user", "assistant"]],
        "input": "instruction",
        "output": "code",
    },
    "dpo": {
        "formats": ["preference"],
        "required": ["prompt", "chosen", "rejected"],
        "input": "prompt",
        "output": "chosen",
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML config files")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def profile_from_config(config_path: Path, profile_id: str | None) -> dict[str, Any]:
    config = load_config(config_path)
    data = config.get("data", {}) or {}
    profiles = data.get("profiles", {}) or {}
    selected = str(profile_id or "").strip()
    if not selected:
        raise ValueError("Dataset profile is required")
    if selected not in profiles:
        raise ValueError(f"Unknown dataset profile: {selected}")
    profile = dict(profiles[selected] or {})
    profile.setdefault("id", selected)
    profile.setdefault("task_type", data.get("task_type", "chatml"))
    profile.setdefault("format", data.get("dataset_format", "chatml_source"))
    return profile


def profiles_from_config(config_path: Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    data = config.get("data", {}) or {}
    profiles = data.get("profiles", {}) or {}
    return [profile_from_config(config_path, profile_id) for profile_id in profiles]


def files_from_profile(profile: dict[str, Any]) -> list[str]:
    return [
        value
        for value in [profile.get("train_file"), profile.get("val_file"), profile.get("test_file")]
        if value
    ]


def extract_source(user: str) -> str:
    if REWRITE_MARKER in user:
        return user.split(REWRITE_MARKER, 1)[1].strip()
    return user.strip()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"{rel(path)}: missing file"]

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel(path)}:{line_no}: invalid JSON: {exc}")
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows, errors


def infer_text(row: dict[str, Any], field: str | None) -> str:
    if not field:
        return ""
    value = row.get(field, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def required_groups(schema: dict[str, Any]) -> list[list[str]]:
    if "required_any" in schema:
        return [list(group) for group in schema["required_any"]]
    return [list(schema.get("required", []))]


def has_required_group(row: dict[str, Any], groups: list[list[str]]) -> bool:
    keys = set(row.keys()) - {"_line_no"}
    return any(all(field in keys for field in group) for group in groups)


def validate_messages(path: Path, row: dict[str, Any], errors: list[str], warnings: list[str]) -> tuple[str, str]:
    line_no = row.get("_line_no", "?")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"{rel(path)}:{line_no}: messages must be a non-empty list")
        return "", ""

    roles = []
    assistant_seen = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{rel(path)}:{line_no}: message {index} must be an object")
            continue
        role = message.get("role")
        roles.append(str(role))
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"{rel(path)}:{line_no}: unsupported role: {role}")
        if role == "assistant":
            assistant_seen = True
            if not message.get("content") and not message.get("tool_calls"):
                errors.append(f"{rel(path)}:{line_no}: assistant message has no content or tool_calls")
    if "user" not in roles:
        errors.append(f"{rel(path)}:{line_no}: messages contain no user turn")
    if not assistant_seen:
        errors.append(f"{rel(path)}:{line_no}: messages contain no assistant turn")
    if roles and roles[0] not in {"system", "user"}:
        warnings.append(f"{rel(path)}:{line_no}: first message role is {roles[0]}")

    first_user = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"), "")
    last_assistant = next(
        (m.get("content", "") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"),
        "",
    )
    return str(first_user or ""), str(last_assistant or "")


def validate_file(
    path: Path,
    task_type: str = "chatml",
    dataset_format: str = "chatml_source",
    min_rows: int = 1,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows, errors = load_jsonl(path)
    warnings: list[str] = []
    quality = quality or {}
    schema = TASK_SCHEMAS.get(task_type)
    if schema is None:
        errors.append(f"{rel(path)}: unknown task_type: {task_type}")
        schema = TASK_SCHEMAS["chatml"]

    if dataset_format not in schema.get("formats", []):
        warnings.append(f"{rel(path)}: format '{dataset_format}' is unusual for task_type '{task_type}'")

    if len(rows) < min_rows:
        errors.append(f"{rel(path)}: expected at least {min_rows} rows, found {len(rows)}")

    pair_keys: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    groups = required_groups(schema)

    label_values = set(quality.get("labels") or quality.get("label_values") or [])
    min_input = int(quality.get("min_input_chars", quality.get("min_source_chars", 1)))
    min_output = int(quality.get("min_output_chars", quality.get("min_target_chars", 1)))
    max_ratio = float(quality.get("max_length_ratio", schema.get("length_ratio_warning", 0)) or 0)

    for row in rows:
        line_no = row.get("_line_no", "?")
        if not has_required_group(row, groups):
            expected = " or ".join("/".join(group) for group in groups)
            errors.append(f"{rel(path)}:{line_no}: missing required field group ({expected})")
            continue

        input_text = ""
        output_text = ""
        if dataset_format == "messages" or "messages" in row:
            input_text, output_text = validate_messages(path, row, errors, warnings)
        elif dataset_format == "chatml_source" or {"user", "assistant"}.issubset(row.keys()):
            user = infer_text(row, "user")
            input_text = extract_source(user) if task_type == "rewrite" else user
            output_text = infer_text(row, "assistant")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            styles[str(metadata.get("style") or metadata.get("author") or "unknown")] += 1
        else:
            input_text = infer_text(row, schema.get("input"))
            output_text = infer_text(row, schema.get("output"))

        if task_type == "classification":
            label = row.get("label", row.get("assistant"))
            labels[str(label)] += 1
            if label_values and str(label) not in label_values:
                errors.append(f"{rel(path)}:{line_no}: label '{label}' is not in configured label set")

        if task_type in {"extraction", "tool_calling"}:
            candidate = row.get("output", row.get("assistant", ""))
            if isinstance(candidate, str) and candidate.strip().startswith(("{", "[")):
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    errors.append(f"{rel(path)}:{line_no}: output is not valid JSON")

        if task_type == "dpo" and row.get("chosen") == row.get("rejected"):
            errors.append(f"{rel(path)}:{line_no}: chosen and rejected are identical")

        text_blob = "\n".join(str(row.get(key, "")) for key in ("system", "user", "assistant", "prompt"))
        for marker in OLD_TASK_MARKERS:
            if marker in text_blob and task_type != "completion":
                warnings.append(f"{rel(path)}:{line_no}: old task marker found: {marker}")

        input_lengths.append(len(input_text))
        output_lengths.append(len(output_text))
        pair_keys[f"{input_text}\n---\n{output_text}"] += 1

        if len(input_text) < min_input:
            warnings.append(f"{rel(path)}:{line_no}: input is short ({len(input_text)} chars)")
        if len(output_text) < min_output:
            warnings.append(f"{rel(path)}:{line_no}: output is short ({len(output_text)} chars)")
        if max_ratio and input_text and output_text:
            ratio = max(len(input_text), len(output_text)) / max(1, min(len(input_text), len(output_text)))
            if ratio > max_ratio:
                warnings.append(f"{rel(path)}:{line_no}: length ratio is high ({ratio:.2f})")

    duplicate_pairs = sum(count - 1 for count in pair_keys.values() if count > 1)
    if duplicate_pairs:
        warnings.append(f"{rel(path)}: duplicate input/output pairs: {duplicate_pairs}")

    return {
        "file": rel(path),
        "task_type": task_type,
        "format": dataset_format,
        "rows": len(rows),
        "styles": dict(styles),
        "labels": dict(labels),
        "input_length": summarize_lengths(input_lengths),
        "output_length": summarize_lengths(output_lengths),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def validate_profile(profile: dict[str, Any], min_rows: int = 1) -> dict[str, Any]:
    task_type = profile.get("task_type", "chatml")
    dataset_format = profile.get("format", profile.get("dataset_format", "chatml_source"))
    quality = profile.get("quality", {}) or {}
    reports = [
        validate_file(resolve(file), task_type, dataset_format, min_rows=min_rows, quality=quality)
        for file in files_from_profile(profile)
        if resolve(file) is not None
    ]
    errors = [msg for report in reports for msg in report["errors"]]
    warnings = [msg for report in reports for msg in report["warnings"]]
    return {
        "profile": profile.get("id", ""),
        "task_type": task_type,
        "format": dataset_format,
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "rows": sum(report["rows"] for report in reports),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local LLM fine-tuning JSONL files.")
    parser.add_argument("files", nargs="*", help="JSONL files to validate.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="YAML config for profile validation.")
    parser.add_argument("--profile", help="Dataset profile id in config.")
    parser.add_argument("--task-type", choices=sorted(TASK_SCHEMAS), help="Override task type.")
    parser.add_argument("--format", dest="dataset_format", help="Override dataset format.")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    if args.files:
        task_type = args.task_type or "chatml"
        dataset_format = args.dataset_format or "chatml_source"
        reports = [
            validate_file(resolve(file), task_type, dataset_format, min_rows=args.min_rows)
            for file in args.files
            if resolve(file) is not None
        ]
        result = {
            "ok": not any(report["errors"] for report in reports),
            "reports": reports,
            "error_count": sum(len(report["errors"]) for report in reports),
            "warning_count": sum(len(report["warnings"]) for report in reports),
        }
    else:
        config_path = resolve(args.config)
        profiles = [profile_from_config(config_path, args.profile)] if args.profile else profiles_from_config(config_path)
        if not profiles:
            raise ValueError("No dataset profiles configured")
        profile_results = []
        for profile in profiles:
            if args.task_type:
                profile["task_type"] = args.task_type
            if args.dataset_format:
                profile["format"] = args.dataset_format
            profile_results.append(validate_profile(profile, min_rows=args.min_rows))
        if len(profile_results) == 1:
            result = profile_results[0]
        else:
            result = {
                "ok": all(item.get("ok") for item in profile_results),
                "profiles": profile_results,
                "error_count": sum(item.get("error_count", 0) for item in profile_results),
                "warning_count": sum(item.get("warning_count", 0) for item in profile_results),
                "rows": sum(item.get("rows", 0) for item in profile_results),
            }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)

    return 0 if result.get("ok") else 1


def print_report(result: dict[str, Any]) -> None:
    print("Data validation")
    print("===============")
    if result.get("profiles"):
        for profile_result in result["profiles"]:
            print_report(profile_result)
        print("\nOverall result:", "OK" if result.get("ok") else "FAILED")
        return
    if result.get("profile"):
        print(f"profile={result['profile']} task_type={result['task_type']} format={result['format']}")
    for report in result.get("reports", []):
        print(f"- {report['file']}: rows={report['rows']}, task_type={report['task_type']}, format={report['format']}")
        if report.get("styles"):
            print(f"  styles={report['styles']}")
        if report.get("labels"):
            print(f"  labels={report['labels']}")
        print(f"  input_length={report['input_length']}")
        print(f"  output_length={report['output_length']}")
    warnings = [msg for report in result.get("reports", []) for msg in report.get("warnings", [])]
    errors = [msg for report in result.get("reports", []) for msg in report.get("errors", [])]
    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors")
        for error in errors:
            print(f"- {error}")
    print("\nResult:", "OK" if result.get("ok") else "FAILED")


if __name__ == "__main__":
    sys.exit(main())
