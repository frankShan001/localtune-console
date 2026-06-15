import json
import inspect

import yaml

from src.services.recipes import export_run_recipe, import_recipe, list_recipes
from src.services.runs import create_run_record
from src import web_dashboard


def test_dashboard_defaults_to_localhost():
    host = inspect.signature(web_dashboard.run_dashboard).parameters["host"].default
    assert host == "127.0.0.1"


def test_recipe_export_and_import_round_trip(tmp_path):
    create_run_record(tmp_path, {
        "id": "run-1",
        "kind": "training",
        "status": "completed",
        "model_id": "qwen27b",
        "branch": "bnb4",
        "dataset_profile": "mini",
        "params": {
            "mode": "smoke",
            "model_id": "qwen27b",
            "branch": "bnb4",
            "dataset_profile": "mini",
            "max_steps": 20,
            "resume_from_checkpoint": "outputs/checkpoint-10",
        },
    })

    exported = export_run_recipe(tmp_path, "run-1", "smoke")
    imported = import_recipe(tmp_path, exported["path"])

    assert imported["payload"]["model_id"] == "qwen27b"
    assert imported["payload"]["max_steps"] == 20
    assert "resume_from_checkpoint" not in imported["payload"]
    assert list_recipes(tmp_path)[0]["name"] == "smoke"


def test_recipe_import_rejects_incomplete_recipe(tmp_path):
    recipe_dir = tmp_path / "examples" / "recipes"
    recipe_dir.mkdir(parents=True)
    path = recipe_dir / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "localtune.recipe.v1",
        "model": {"id": "qwen27b"},
    }), encoding="utf-8")

    try:
        import_recipe(tmp_path, str(path))
    except ValueError as exc:
        assert "missing model" in str(exc)
    else:
        raise AssertionError("Incomplete recipe should be rejected")


def test_corpus_preview_supports_search_and_pagination(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    corpus = data_dir / "train.jsonl"
    corpus.write_text("\n".join(
        json.dumps({"instruction": "rewrite", "input": f"item {index}", "output": f"result {index}"})
        for index in range(6)
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", tmp_path)

    page = web_dashboard.read_corpus_preview("data/train.jsonl", limit=2, offset=2)
    searched = web_dashboard.read_corpus_preview("data/train.jsonl", limit=10, query="item 4")

    assert [sample["line"] for sample in page["samples"]] == [3, 4]
    assert page["pagination"]["total"] == 6
    assert page["pagination"]["has_next"]
    assert searched["pagination"]["total"] == 1
    assert searched["samples"][0]["line"] == 5


def test_dashboard_exposes_core_workflows():
    routes = {rule.rule for rule in web_dashboard.app.url_map.iter_rules()}
    assert {
        "/api/datasets/profiles",
        "/api/training/start",
        "/api/training/status",
        "/api/golden-path/status",
        "/api/golden-path/plan",
        "/api/environment/repair",
        "/api/artifacts/manage",
        "/api/inference/base-model",
        "/api/inference/run",
        "/api/inference/batch",
        "/api/environment/dependencies",
        "/api/model-exports",
        "/api/model-exports/start",
        "/api/models/select-directory",
        "/api/corpus/import",
        "/api/recipes",
        "/api/recipes/export",
        "/api/recipes/import",
    }.issubset(routes)
    assert "/api/inference/default-target" not in routes
    assert web_dashboard.app.test_client().get("/api/inference/default-target").status_code == 404


def test_empty_model_config_does_not_create_placeholder_models():
    active_model, models = web_dashboard.get_model_catalog_summary(
        {"active_model": "", "name": "", "models": {}},
        {"bnb4": {"model_path": "", "quant_type": "nf4"}},
    )

    assert active_model == ""
    assert models == []
    assert web_dashboard.get_model_scan_dirs({"scan_dirs": []}) == []


def test_model_directory_picker_api_returns_selected_path(monkeypatch, tmp_path):
    monkeypatch.setattr(web_dashboard, "choose_local_directory", lambda _initial=None: str(tmp_path))

    response = web_dashboard.app.test_client().post("/api/models/select-directory", json={})

    assert response.status_code == 200
    assert response.get_json() == {"cancelled": False, "path": str(tmp_path)}


def test_import_corpus_file_copies_jsonl_and_creates_profile(tmp_path, monkeypatch):
    source = tmp_path / "incoming.jsonl"
    source.write_text(
        json.dumps({"instruction": "rewrite", "input": "plain", "output": "styled"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / "model_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        "data": {
            "task_type": "rewrite",
            "dataset_format": "chatml_source",
            "profiles": {},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_dashboard, "BASE_CONFIG_PATH", config_path)
    monkeypatch.setattr(web_dashboard, "CONFIG_STORE", web_dashboard.ProjectConfigStore(config_path))

    result = web_dashboard.import_corpus_file({"path": str(source)})

    imported = tmp_path / result["file"]["path"]
    assert result["cancelled"] is False
    assert result["profile"] == "incoming"
    assert imported.exists()
    assert imported.parent == tmp_path / "data" / "processed"
    profiles = web_dashboard.get_dataset_profiles()
    assert profiles[0]["id"] == "incoming"
    assert profiles[0]["train"]["rows"] == 1


def test_environment_dependencies_exposes_structured_status():
    response = web_dashboard.app.test_client().get("/api/environment/dependencies")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["items"]
    assert payload["counts"]["ready"] >= 1
    assert {"id", "name", "version", "requirement", "required", "status", "detail"}.issubset(payload["items"][0])
    assert {"mode", "hint"}.issubset(payload["items"][0]["repair"])


def test_environment_repair_route_returns_action_output(monkeypatch):
    monkeypatch.setattr(web_dashboard, "repair_environment_dependencies", lambda: {
        "ok": True,
        "stdout": "[setup] repaired",
        "stderr": "",
        "profile": {"backend": "cuda"},
        "dependencies": {"items": [], "counts": {}},
    })

    response = web_dashboard.app.test_client().post("/api/environment/repair", json={})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert "repaired" in payload["stdout"]


def test_metrics_api_reads_localtune_jsonl(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    run_dir = output_root / "localtune" / "bnb4" / "runs" / "job-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "\n".join([
            json.dumps({"step": 1, "wall_time": 10.0, "metrics": {"loss": 5.8, "learning_rate": 2e-4}}),
            json.dumps({"step": 2, "wall_time": 11.0, "metrics": {"loss": 5.6, "eval_loss": 5.7}}),
        ]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_dashboard, "OUTPUTS_DIR", output_root)

    payload = web_dashboard.get_training_metrics("job-1")

    assert payload["source"].replace("\\", "/") == "outputs/localtune/bnb4/runs/job-1/metrics.jsonl"
    assert [item["value"] for item in payload["train_loss"]] == [5.8, 5.6]
    assert payload["latest"]["train_loss"]["step"] == 2
    assert payload["latest"]["learning_rate"]["value"] == 2e-4


def test_golden_path_exposes_training_readiness_contract():
    response = web_dashboard.app.test_client().get("/api/golden-path/status")
    payload = response.get_json()
    readiness = payload["training_readiness"]
    guidance = payload["model_guidance"]

    assert response.status_code == 200
    assert readiness["code"] in {
        "ready",
        "running",
        "unsupported_backend",
        "missing_environment",
        "missing_model",
        "missing_dataset",
    }
    assert readiness["route"] in {"golden", "training", "environment", "models", "corpus"}
    assert isinstance(readiness["can_train"], bool)
    assert {"status", "backend", "load_method", "available_vram_gb"}.issubset(guidance)


def test_golden_training_dependency_counts_ignore_frontend_toolchain():
    dependencies = {
        "items": [
            {"id": "python", "required": True, "status": "ready"},
            {"id": "torch", "required": True, "status": "ready"},
            {"id": "bitsandbytes", "required": True, "status": "ready"},
            {"id": "node", "required": True, "status": "missing"},
            {"id": "npm", "required": True, "status": "missing"},
            {"id": "unsloth", "required": False, "status": "optional"},
        ]
    }

    assert web_dashboard._training_dependency_counts(dependencies) == {
        "required_total": 3,
        "required_ready": 3,
    }


def test_golden_path_tolerates_empty_package_version_metadata(monkeypatch):
    real_version = web_dashboard.importlib.metadata.version
    real_import_module = web_dashboard.importlib.import_module

    def fake_version(distribution):
        if distribution == "torch":
            return None
        return real_version(distribution)

    def fake_import_module(distribution):
        if distribution == "torch":
            return type("TorchModule", (), {"__version__": "2.12.0+cu130"})()
        return real_import_module(distribution)

    monkeypatch.setattr(web_dashboard.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(web_dashboard.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(web_dashboard, "_torch_accelerator_info", lambda: {
        "available": False,
        "backend": "cpu",
        "device_name": "CPU",
        "device_count": 0,
        "message": "No supported accelerator backend available",
        "source": "test",
    })
    with web_dashboard._GOLDEN_PATH_CACHE_LOCK:
        web_dashboard._GOLDEN_PATH_CACHE.clear()

    dependencies = web_dashboard.get_environment_dependencies()
    torch_item = next(item for item in dependencies["items"] if item["id"] == "torch")
    response = web_dashboard.app.test_client().get("/api/golden-path/status")

    assert torch_item["version"] == "2.12.0+cu130"
    assert torch_item["status"] == "ready"
    assert response.status_code == 200
    assert {"score", "steps", "next_step", "payloads"}.issubset(response.get_json())


def test_training_start_returns_stable_error_code():
    response = web_dashboard.app.test_client().post("/api/training/start", json={})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["code"] == "BASE_MODEL_REQUIRED"
    assert payload["error"]


def test_core_api_contracts_are_stable():
    client = web_dashboard.app.test_client()
    contracts = {
        "/api/training/status": {"status", "job"},
        "/api/datasets": {"profiles"},
        "/api/artifacts": {"root", "items"},
        "/api/runs": {"runs"},
        "/api/golden-path/status": {"score", "steps", "next_step", "payloads"},
        "/api/models/recommendations": {"recommendations", "provider", "accelerator"},
        "/api/models/downloads": {"items"},
    }

    for path, required_keys in contracts.items():
        response = client.get(path)
        assert response.status_code == 200, path
        assert required_keys.issubset(response.get_json()), path


def test_dashboard_port_supports_environment_override(monkeypatch):
    monkeypatch.setenv("LOCALTUNE_PORT", "7654")
    monkeypatch.setenv("LOCALTUNE_TENSORBOARD_PORT", "7007")

    assert web_dashboard.load_monitoring_config() == 7654


def test_unix_launcher_is_available():
    launcher = web_dashboard.PROJECT_ROOT / "start_localtune.sh"
    assert launcher.exists()
    assert "scripts/start_dashboard.py" in launcher.read_text(encoding="utf-8")
