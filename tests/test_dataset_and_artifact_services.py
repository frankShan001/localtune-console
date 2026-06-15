import json
import threading
from pathlib import Path

import yaml

from scripts.validate_data import validate_file
from src.services.errors import LocalTuneError
from src.services.artifacts import inspect_adapter
from src.services.config_store import ProjectConfigStore
from src.services.datasets import build_dataset_profiles
from src.services.runs import create_run_record, read_run_record
from src.services.training_jobs import TrainingManager


def test_dataset_profiles_are_explicit_and_include_split_metadata(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"instruction":"rewrite","input":"a","output":"b"}\n', encoding="utf-8")

    assert build_dataset_profiles(tmp_path, {}) == []

    profiles = build_dataset_profiles(tmp_path, {
        "task_type": "rewrite",
        "dataset_format": "localtune_v1",
        "profiles": {
            "mini": {
                "name": "Mini",
                "train_file": "train.jsonl",
            }
        },
    })

    assert len(profiles) == 1
    assert profiles[0].id == "mini"
    assert profiles[0].files["train"].rows == 1
    assert profiles[0].files["val"] is None


def test_typical_corpus_examples_match_their_declared_schemas():
    examples = [
        ("instruction.jsonl", "instruction", "alpaca"),
        ("rewrite.jsonl", "rewrite", "source_target"),
        ("chat.jsonl", "chat", "messages"),
        ("dpo.jsonl", "dpo", "preference"),
    ]

    for filename, task_type, dataset_format in examples:
        result = validate_file(
            Path("examples/corpus") / filename,
            task_type=task_type,
            dataset_format=dataset_format,
            min_rows=1,
        )
        assert result["ok"], result["errors"]


def test_adapter_inspection_reports_required_files(tmp_path):
    missing = inspect_adapter(tmp_path)
    assert not missing["ok"]
    assert len(missing["errors"]) == 2

    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen3.6-27B",
        "target_modules": ["q_proj"],
        "r": 16,
        "lora_alpha": 32,
    }), encoding="utf-8")

    result = inspect_adapter(tmp_path)
    assert result["ok"]
    assert result["base_model"] == "Qwen/Qwen3.6-27B"
    assert result["r"] == 16


def test_failure_diagnostics_identifies_cuda_oom(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory", encoding="utf-8")

    result = TrainingManager._failure_diagnostics(None, log)

    assert result["code"] == "cuda_oom"
    assert len(result["suggestions"]) >= 3


def test_project_config_store_serializes_atomic_updates(tmp_path):
    config_path = tmp_path / "configs" / "model_config.yaml"
    store = ProjectConfigStore(config_path)
    store.write({"counter": 0})

    def increment():
        for _ in range(20):
            store.update(lambda config: config.update(counter=config.get("counter", 0) + 1))

    threads = [threading.Thread(target=increment) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.read()["counter"] == 80
    assert not list(config_path.parent.glob("*.tmp"))


def make_training_manager(tmp_path, config_path):
    return TrainingManager(
        tmp_path,
        config_path,
        tmp_path / "configs" / "runtime",
        tmp_path / "logs",
        tmp_path / "outputs",
        lambda value: Path(value) if Path(value).is_absolute() else tmp_path / value,
        lambda value: str(Path(value).resolve().relative_to(tmp_path.resolve())),
    )


def test_training_manager_uses_runtime_output_directory(tmp_path):
    config_path = tmp_path / "configs" / "model_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        "training": {"output_dir": "./custom-output"},
    }), encoding="utf-8")
    manager = make_training_manager(tmp_path, config_path)

    output_dir = manager._output_dir_from_config(config_path, "bnb4")

    assert output_dir == tmp_path / "custom-output" / "bnb4"


def test_training_manager_rejects_nvfp4_training_branch(tmp_path):
    config_path = tmp_path / "configs" / "model_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        "quantization": {
            "branches": {
                "bnb4": {"load_mode": "bnb_qlora", "quant_type": "nf4"},
                "nvfp4": {"load_mode": "nvfp4_qlora", "quant_type": "nvfp4"},
            }
        },
    }), encoding="utf-8")
    manager = make_training_manager(tmp_path, config_path)

    try:
        manager._validate_branch_backend("nvfp4")
    except LocalTuneError as exc:
        assert exc.code == "BRANCH_UNSUPPORTED"
    else:
        raise AssertionError("NVFP4 branch should be rejected before training starts")


def test_training_manager_marks_stale_runs_interrupted(tmp_path):
    config_path = tmp_path / "configs" / "model_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("training: {}\n", encoding="utf-8")
    create_run_record(tmp_path, {
        "id": "stale-run",
        "kind": "training",
        "status": "running",
    })

    make_training_manager(tmp_path, config_path)

    record = read_run_record(tmp_path, "stale-run")
    assert record["status"] == "interrupted"
    assert record["diagnostics"]["code"] == "dashboard_restarted"
