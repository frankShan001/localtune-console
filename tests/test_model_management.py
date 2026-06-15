import json

import yaml

from src import web_dashboard
from src.services.errors import LocalTuneError


def write_model_config(path, branches):
    config_path = path / "configs" / "model_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        "quantization": {
            "active_branch": "bnb4",
            "branches": branches,
        },
        "model": {
            "active_model": "",
            "models": {},
            "scan_dirs": [],
        },
    }), encoding="utf-8")
    return config_path


def write_hf_model(path, *, model_type="qwen3", quantization_config=None):
    path.mkdir(parents=True)
    payload = {"model_type": model_type, "architectures": ["Qwen3ForCausalLM"]}
    if quantization_config:
        payload["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"not a real model")


def use_project(monkeypatch, root, config_path):
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", root)
    monkeypatch.setattr(web_dashboard, "BASE_CONFIG_PATH", config_path)
    monkeypatch.setattr(web_dashboard, "CONFIG_STORE", web_dashboard.ProjectConfigStore(config_path))


def write_artifact(path, *, artifact_type="final_adapter", branch="bnb4", best=False):
    path.mkdir(parents=True)
    if artifact_type in {"final_adapter", "checkpoint"}:
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
        (path / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen3.6-27B",
            "target_modules": ["q_proj"],
            "r": 16,
            "lora_alpha": 32,
        }), encoding="utf-8")
    web_dashboard.write_artifact_manifest(path, {
        "artifact": {
            "type": artifact_type,
            "branch": branch,
            "path": str(path),
        },
        "run": {
            "id": "run-1",
            "dataset_profile": "mini",
        },
        "paths": {
            "artifact": str(path),
        },
        "management": {
            "best": best,
        },
    })


def test_model_detection_ignores_project_root_quantization_words(tmp_path, monkeypatch):
    project = tmp_path / "nvfp4-qlora"
    model_dir = project / "models" / "Qwen" / "Qwen3___6-27B"
    write_hf_model(model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
        "unsloth": {"quant_type": "nf4", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)

    candidate = web_dashboard.detect_model_directory(model_dir)
    result = web_dashboard.register_model_candidate({"candidate": candidate, "make_active": True})

    assert candidate["quant_format"] == "base"
    assert result["ok"] is True
    assert set(result["model"]["paths"]) == {"bnb4", "unsloth"}
    assert all(
        value.replace("\\", "/") == "models/Qwen/Qwen3___6-27B"
        for value in result["model"]["paths"].values()
    )
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["active_model"] == result["model_id"]


def test_model_suitability_estimates_size_from_model_name():
    accelerator = {"backend": "cuda", "max_memory": 23.9, "device_name": "RTX 5090 Laptop GPU"}

    small = web_dashboard.model_hardware_suitability({"name": "Qwen/Qwen2.5-14B"}, accelerator)
    mid = web_dashboard.model_hardware_suitability({"name": "Qwen/Qwen3.6-27B"}, accelerator)
    large = web_dashboard.model_hardware_suitability({"name": "Qwen/Qwen3.6-35B"}, accelerator)

    assert small["params_b"] == 14
    assert small["status"] == "recommended"
    assert mid["status"] == "caution"
    assert large["status"] == "not_recommended"


def test_model_guidance_uses_common_model_size_buckets():
    guidance = web_dashboard.model_guidance_for_accelerator({
        "backend": "cuda",
        "max_memory": 23.9,
        "device_name": "RTX 5090 Laptop GPU",
    })

    assert guidance["recommended_max_params_b"] == 14
    assert guidance["caution_max_params_b"] == 27


def test_model_recommendations_use_locale_provider_and_vram_fit(monkeypatch):
    monkeypatch.setattr(web_dashboard, "_cached_golden_check", lambda *_args: {
        "backend": "cuda",
        "max_memory": 23.9,
        "device_name": "RTX 5090 Laptop GPU",
    })
    monkeypatch.setattr(web_dashboard, "list_model_download_jobs", lambda: {"items": []})

    zh = web_dashboard.get_model_recommendations("zh")
    en = web_dashboard.get_model_recommendations("en")

    assert zh["provider"] == "modelscope"
    assert en["provider"] == "huggingface"
    assert zh["recommendations"][0]["family"] == "Qwen"
    assert en["recommendations"][0]["family"] == "Gemma"
    qwen9 = next(item for item in zh["recommendations"] if item["id"] == "qwen3_5_9b")
    qwen27 = next(item for item in zh["recommendations"] if item["id"] == "qwen3_6_27b")
    assert qwen9["fit"]["status"] == "recommended"
    assert qwen27["fit"]["status"] == "caution"
    assert qwen9["download_url"].startswith("https://modelscope.cn/models/")
    assert next(item for item in en["recommendations"] if item["id"] == "qwen3_5_9b")["download_url"].startswith("https://huggingface.co/")


def test_cancel_model_download_kills_orphaned_child_process(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "model_downloads.json"
    monkeypatch.setattr(web_dashboard, "RUNTIME_CONFIG_DIR", state_path.parent)
    monkeypatch.setattr(web_dashboard, "MODEL_DOWNLOAD_STATE_PATH", state_path)
    monkeypatch.setattr(web_dashboard, "_MODEL_DOWNLOAD_PROCESSES", {})
    web_dashboard._write_download_state_items([{
        "id": "job1",
        "model_id": "qwen3_5_4b",
        "model_name": "Qwen/Qwen3.5-4B",
        "provider": "modelscope",
        "status": "failed",
        "pid": 12345,
        "target_dir": "models",
        "log_file": "logs/model_download_job1.log",
    }])

    monkeypatch.setattr(web_dashboard, "_model_download_processes_for_job", lambda item: [object()])
    monkeypatch.setattr(web_dashboard, "_terminate_model_download_processes", lambda item: 2)

    result = web_dashboard.cancel_model_download("job1")

    assert result["ok"] is True
    item = result["downloads"][0]
    assert item["status"] == "cancelled"
    assert item["killed_processes"] == 2
    assert item["returncode"] == -15


def test_refresh_download_marks_completed_from_log_when_parent_exited(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "model_downloads.json"
    log_path = tmp_path / "logs" / "model_download_job1.log"
    model_dir = tmp_path / "models" / "Qwen" / "Qwen3___5-4B"
    write_hf_model(model_dir)
    log_path.parent.mkdir(parents=True)
    log_path.write_text("[YES] Download completed\nModel path: models/Qwen/Qwen3___5-4B\n", encoding="utf-8")
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_dashboard, "RUNTIME_CONFIG_DIR", state_path.parent)
    monkeypatch.setattr(web_dashboard, "MODEL_DOWNLOAD_STATE_PATH", state_path)
    monkeypatch.setattr(web_dashboard, "_MODEL_DOWNLOAD_PROCESSES", {})
    monkeypatch.setattr(web_dashboard, "_model_download_processes_for_job", lambda item: [])
    web_dashboard._write_download_state_items([{
        "id": "job1",
        "model_id": "qwen3_5_4b",
        "model_name": "Qwen/Qwen3.5-4B",
        "provider": "modelscope",
        "status": "failed",
        "pid": 12345,
        "target_dir": "models",
        "log_file": "logs/model_download_job1.log",
        "returncode": 1,
    }])

    items = web_dashboard.list_model_download_jobs()["items"]

    assert items[0]["status"] == "completed"
    assert items[0]["returncode"] == 0
    assert web_dashboard.resolve_project_path(items[0]["model_path"]) == model_dir


def test_refresh_download_marks_missing_when_completed_model_was_removed(tmp_path, monkeypatch):
    state_path = tmp_path / "runtime" / "model_downloads.json"
    log_path = tmp_path / "logs" / "model_download_job1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("[YES] Download completed\nModel path: models/Qwen/Qwen3___5-4B\n", encoding="utf-8")
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_dashboard, "RUNTIME_CONFIG_DIR", state_path.parent)
    monkeypatch.setattr(web_dashboard, "MODEL_DOWNLOAD_STATE_PATH", state_path)
    monkeypatch.setattr(web_dashboard, "_MODEL_DOWNLOAD_PROCESSES", {})
    monkeypatch.setattr(web_dashboard, "_model_download_processes_for_job", lambda item: [])
    web_dashboard._write_download_state_items([{
        "id": "job1",
        "model_id": "qwen3_5_4b",
        "model_name": "Qwen/Qwen3.5-4B",
        "provider": "modelscope",
        "status": "completed",
        "pid": 12345,
        "target_dir": "models",
        "model_path": "models/Qwen/Qwen3___5-4B",
        "log_file": "logs/model_download_job1.log",
        "returncode": 0,
    }])

    items = web_dashboard.list_model_download_jobs()["items"]

    assert items[0]["status"] == "missing"
    assert items[0]["error"] == "downloaded_model_missing"
    assert "model_path" not in items[0]


def test_scan_candidate_includes_hardware_suitability(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_dir = project / "models" / "Qwen" / "Qwen3___6-27B"
    write_hf_model(model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)
    monkeypatch.setattr(web_dashboard, "_cached_golden_check", lambda *_args: {
        "backend": "cuda",
        "max_memory": 23.9,
        "device_name": "RTX 5090 Laptop GPU",
    })

    result = web_dashboard.scan_model_directory(project / "models")

    suitability = result["candidates"][0]["suitability"]
    assert suitability["params_b"] == 27
    assert suitability["status"] == "caution"
    assert suitability["available_vram_gb"] == 23.9


def test_config_summary_hides_unsupported_nvfp4_branch(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_dir = project / "models" / "Qwen" / "Qwen3___6-27B"
    write_hf_model(model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "load_mode": "bnb_qlora", "model_path": "models/Qwen/Qwen3___6-27B"},
        "nvfp4": {"quant_type": "nvfp4", "load_mode": "nvfp4_qlora", "model_path": "models/Qwen/Qwen3___6-27B-NVFP4"},
    })
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["active_model"] = "qwen"
    config["model"]["models"] = {
        "qwen": {
            "name": "Qwen",
            "paths": {
                "bnb4": "models/Qwen/Qwen3___6-27B",
                "nvfp4": "models/Qwen/Qwen3___6-27B-NVFP4",
            },
        }
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    use_project(monkeypatch, project, config_path)
    monkeypatch.setattr(web_dashboard, "_cached_golden_check", lambda *_args: {"backend": "cuda"})

    summary = web_dashboard.get_project_config_summary()

    assert [branch["id"] for branch in summary["branches"]] == ["bnb4"]
    assert [branch["id"] for branch in summary["models"][0]["branches"]] == ["bnb4"]


def test_scan_model_directory_skips_project_internal_test_dirs(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    test_model_dir = project / ".pytest-tmp" / "test_config_summary_hides_unsu0" / "LocalTune" / "models" / "Qwen" / "Qwen3___6-27B"
    write_hf_model(test_model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "load_mode": "bnb_qlora", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)

    result = web_dashboard.scan_model_directory(project)

    assert result["candidates"] == []


def test_remove_model_catalog_item_removes_stale_registered_model(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_dir = project / "models" / "Qwen" / "Qwen3___6-27B"
    write_hf_model(model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "load_mode": "bnb_qlora", "model_path": ""},
    })
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["active_model"] = "qwen"
    config["model"]["models"] = {
        "qwen": {
            "name": "Qwen",
            "paths": {"bnb4": "models/Qwen/Qwen3___6-27B"},
        }
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    use_project(monkeypatch, project, config_path)

    result = web_dashboard.remove_model_catalog_item("qwen")
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert saved["model"]["models"] == {}
    assert saved["model"]["active_model"] == ""
    assert model_dir.exists()


def test_nvfp4_model_requires_compatible_branch(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_dir = project / "models" / "Qwen" / "Qwen3___6-27B-NVFP4"
    write_hf_model(model_dir, quantization_config={"quant_method": "nvfp4"})
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)

    candidate = web_dashboard.detect_model_directory(model_dir)

    assert candidate["quant_format"] == "nvfp4"
    try:
        web_dashboard.register_model_candidate({"candidate": candidate})
    except LocalTuneError as exc:
        assert exc.code == "MODEL_BRANCH_UNSUPPORTED"
    else:
        raise AssertionError("NVFP4 model should require an NVFP4-compatible branch")


def test_register_model_api_returns_specific_error_code(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)

    response = web_dashboard.app.test_client().post(
        "/api/models/register",
        json={"candidate": {"path": str(project / "missing-model")}},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert payload["code"] == "MODEL_DIRECTORY_UNRECOGNIZED"
    assert payload["error"]


def test_scan_reports_gguf_as_unsupported_training_input(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_root = project / "models"
    gguf_dir = model_root / "Qwen-GGUF"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "qwen.gguf").write_bytes(b"gguf")
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    use_project(monkeypatch, project, config_path)

    result = web_dashboard.scan_model_directory(model_root)

    assert result["candidates"] == []
    assert result["notices"][0]["format"] == "gguf"
    assert result["notices"][0]["file_count"] == 1


def test_artifact_list_excludes_metrics_runs(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"] = {"output_dir": "./outputs"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    use_project(monkeypatch, project, config_path)
    write_artifact(project / "outputs" / "bnb4" / "final", artifact_type="final_adapter")
    write_artifact(project / "outputs" / "bnb4" / "runs" / "run-1", artifact_type="metrics_run")

    result = web_dashboard.list_artifacts()

    assert [item["type"] for item in result["items"]] == ["final_adapter"]


def test_can_unmark_best_artifact(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": ""},
    })
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"] = {"output_dir": "./outputs"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    use_project(monkeypatch, project, config_path)
    artifact_path = project / "outputs" / "bnb4" / "final"
    write_artifact(artifact_path, best=True)

    response = web_dashboard.manage_artifact({"action": "unbest", "path": "outputs/bnb4/final"})

    assert response["ok"]
    manifest = web_dashboard.read_artifact_manifest(artifact_path)
    assert manifest["management"]["best"] is False


def test_model_export_starts_merge_with_local_base_model(tmp_path, monkeypatch):
    project = tmp_path / "LocalTune"
    model_dir = project / "models" / "Qwen3_6_27B"
    write_hf_model(model_dir)
    config_path = write_model_config(project, {
        "bnb4": {"quant_type": "nf4", "model_path": str(model_dir)},
    })
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"] = {"output_dir": "./outputs"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    use_project(monkeypatch, project, config_path)
    monkeypatch.setattr(web_dashboard, "LOGS_DIR", project / "logs")
    monkeypatch.setattr(web_dashboard, "OUTPUTS_DIR", project / "outputs")
    monkeypatch.setattr(web_dashboard, "RUNTIME_CONFIG_DIR", project / "configs" / "runtime")
    monkeypatch.setattr(web_dashboard, "MODEL_EXPORT_STATE_PATH", project / "configs" / "runtime" / "model_exports.json")
    monkeypatch.setattr(web_dashboard, "_MODEL_EXPORT_PROCESSES", {})

    artifact_path = project / "outputs" / "bnb4" / "final"
    write_artifact(artifact_path)
    (artifact_path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "base_model_name_or_path": str(model_dir),
        "target_modules": ["q_proj"],
        "r": 16,
        "lora_alpha": 32,
    }), encoding="utf-8")

    created_processes = []

    class FakeProcess:
        def __init__(self):
            self.pid = 4242
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    def fake_popen(command, **_kwargs):
        process = FakeProcess()
        process.command = command
        created_processes.append(process)
        return process

    monkeypatch.setattr(web_dashboard.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(web_dashboard, "_processes_for_export_job", lambda _item: [])

    result = web_dashboard.start_model_export({
        "adapter_path": "outputs/bnb4/final",
        "dtype": "fp16",
    })

    assert result["ok"] is True
    job = result["job"]
    command = created_processes[0].command
    assert command[command.index("--base_model") + 1] == str(model_dir)
    assert command[command.index("--lora_path") + 1] == str(artifact_path)
    assert job["status"] == "running"

    output_path = web_dashboard.resolve_project_path(job["output_path"])
    output_path.mkdir(parents=True)
    created_processes[0].returncode = 0
    jobs = web_dashboard.list_model_export_jobs()["items"]
    completed = next(item for item in jobs if item["id"] == job["id"])
    assert completed["status"] == "completed"
    manifest = web_dashboard.read_artifact_manifest(output_path)
    assert manifest["artifact"]["type"] == "merged_model"
    assert web_dashboard.resolve_project_path(manifest["source"]["adapter_path"]) == artifact_path
