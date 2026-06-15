import os
from pathlib import Path

from scripts import prepare_project
from scripts.check_release_payload import release_payload_violations


def test_ensure_local_config_copies_example(tmp_path, monkeypatch):
    example = tmp_path / "configs" / "model_config.example.yaml"
    local = tmp_path / "configs" / "model_config.yaml"
    example.parent.mkdir(parents=True)
    example.write_text("project:\n  name: localtune-console\n", encoding="utf-8")

    monkeypatch.setattr(prepare_project, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_project, "CONFIG_EXAMPLE_PATH", example)
    monkeypatch.setattr(prepare_project, "CONFIG_PATH", local)

    assert prepare_project.ensure_local_config()
    assert local.read_text(encoding="utf-8") == example.read_text(encoding="utf-8")
    assert not prepare_project.ensure_local_config()


def test_run_exposes_local_tool_directory_to_child_processes(tmp_path, monkeypatch):
    tool = tmp_path / "tools" / "npm.cmd"
    tool.parent.mkdir()
    tool.write_text("", encoding="utf-8")
    captured = {}

    class Result:
        returncode = 0

    def fake_run(command, cwd, env):
        captured.update(command=command, cwd=cwd, env=env)
        return Result()

    monkeypatch.setattr(prepare_project.subprocess, "run", fake_run)

    prepare_project.run([str(tool), "ci"], tmp_path)

    assert captured["cwd"] == tmp_path
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(Path(tool).resolve().parent)


def test_frontend_toolchain_installs_into_project_venv(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    scripts = venv / ("Scripts" if prepare_project.sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    calls = []
    node_suffix = ".exe" if prepare_project.sys.platform == "win32" else ""
    npm_suffix = ".cmd" if prepare_project.sys.platform == "win32" else ""
    node = scripts / f"node{node_suffix}"
    npm = scripts / f"npm{npm_suffix}"

    monkeypatch.setattr(prepare_project, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_project.sys, "prefix", str(venv))
    monkeypatch.setattr(prepare_project, "frontend_toolchain_candidates", lambda: [])
    monkeypatch.setattr(
        prepare_project,
        "local_executable",
        lambda name: node if name == "node" else npm,
    )
    monkeypatch.setattr(
        prepare_project,
        "validate_frontend_toolchain",
        lambda node_path, npm_path: (True, "Node.js v22.14.0, npm 10.9.2"),
    )
    monkeypatch.setattr(
        prepare_project,
        "run",
        lambda command, cwd, extra_path=None: calls.append((command, cwd)),
    )

    node_path, npm_path = prepare_project.ensure_frontend_toolchain()

    assert node_path == str(node)
    assert npm_path == str(npm)
    assert calls[0][0][:3] == [prepare_project.sys.executable, "-m", "nodeenv"]
    assert "--python-virtualenv" in calls[0][0]
    assert calls[0][1] == tmp_path


def test_parse_major_version():
    assert prepare_project.parse_major_version("v22.14.0") == 22
    assert prepare_project.parse_major_version("10.9.2") == 10
    assert prepare_project.parse_major_version("unknown") is None


def test_frontend_toolchain_rejects_old_node(monkeypatch):
    versions = {
        "node": "v20.18.0",
        "npm": "10.8.2",
    }
    monkeypatch.setattr(
        prepare_project,
        "command_version",
        lambda command, extra_path=None: versions[Path(command).stem],
    )

    valid, detail = prepare_project.validate_frontend_toolchain("node", "npm")

    assert not valid
    assert "22+ is required" in detail


def test_frontend_toolchain_rejects_old_npm(monkeypatch):
    versions = {
        "node": "v22.14.0",
        "npm": "9.9.4",
    }
    monkeypatch.setattr(
        prepare_project,
        "command_version",
        lambda command, extra_path=None: versions[Path(command).stem],
    )

    valid, detail = prepare_project.validate_frontend_toolchain("node", "npm")

    assert not valid
    assert "10+ is required" in detail


def test_release_payload_rejects_runtime_and_model_files():
    violations = release_payload_violations([
        "src/config.py",
        "examples/corpus/rewrite.jsonl",
        "logs/train.log",
        "configs/runtime/job.yaml",
        "models/model.safetensors",
    ])

    assert violations == [
        "configs/runtime/job.yaml",
        "logs/train.log",
        "models/model.safetensors",
    ]
