"""Training job process management for the LocalTune web console."""

from __future__ import annotations

import os
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from src.constants import DEFAULT_OUTPUT_DIR
from src.services.artifacts import SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION, write_artifact_manifest
from src.services.errors import LocalTuneError
from src.services.runs import create_run_record, list_run_records, update_run_record

logger = logging.getLogger(__name__)


class TrainingManager:
    """Owns at most one local training subprocess."""

    def __init__(
        self,
        project_root: Path,
        base_config_path: Path,
        runtime_config_dir: Path,
        logs_dir: Path,
        outputs_dir: Path,
        resolve_project_path,
        relative_path,
        runtime_config_keep: int = 10,
    ):
        self.project_root = project_root
        self.base_config_path = base_config_path
        self.runtime_config_dir = runtime_config_dir
        self.logs_dir = logs_dir
        self.outputs_dir = outputs_dir
        self.resolve_project_path = resolve_project_path
        self.relative_path = relative_path
        self.runtime_config_keep = runtime_config_keep
        self.lock = threading.Lock()
        self.process = None
        self.job = None
        self.reader_thread = None
        self._recover_interrupted_runs()

    def start(self, payload):
        with self.lock:
            payload = dict(payload or {})
            self._refresh_locked()
            if self.process and self.process.poll() is None:
                raise LocalTuneError("TRAINING_ALREADY_RUNNING", "A training job is already running", 409)

            job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            mode = payload.get("mode", "smoke")
            branch = payload.get("branch", "bnb4")
            model_id = payload.get("model_id", "")
            dataset_profile = payload.get("dataset_profile", "")
            no_fallback = bool(payload.get("no_fallback", True))
            if not model_id:
                raise LocalTuneError("BASE_MODEL_REQUIRED", "A base model must be selected")
            if not dataset_profile:
                raise LocalTuneError("DATASET_PROFILE_REQUIRED", "A dataset profile must be selected")
            self._validate_branch_backend(branch)
            resume_from_checkpoint = str(payload.get("resume_from_checkpoint") or "").strip()
            if resume_from_checkpoint:
                checkpoint_path = self.resolve_project_path(resume_from_checkpoint)
                if not checkpoint_path or not checkpoint_path.is_dir() or not checkpoint_path.name.startswith("checkpoint-"):
                    raise LocalTuneError("CHECKPOINT_INVALID", "Select a valid training checkpoint")
                resume_from_checkpoint = self.relative_path(checkpoint_path)
                payload["resume_from_checkpoint"] = resume_from_checkpoint

            config_path = self._write_runtime_config(job_id, mode, payload)
            output_dir = self._output_dir_from_config(config_path, branch)
            log_path = self.logs_dir / f"web_train_{job_id}.log"
            self.logs_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-u",
                str(self.project_root / "scripts" / "train.py"),
                "--config",
                str(config_path.relative_to(self.project_root)),
                "--branch",
                branch,
            ]
            if no_fallback:
                cmd.append("--no-fallback")
            if resume_from_checkpoint:
                cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["LOCALTUNE_JOB_ID"] = job_id
            env.setdefault("WANDB_DISABLED", "true")

            started_at = datetime.now()
            log_file = open(log_path, "a", encoding="utf-8", errors="ignore")
            log_file.write(f"[dashboard] job_id={job_id}\n")
            log_file.write(f"[dashboard] mode={mode}, model={model_id or 'config'}, branch={branch}\n")
            log_file.write(f"[dashboard] dataset_profile={dataset_profile or 'default'}\n")
            log_file.write("[dashboard] command=" + " ".join(cmd) + "\n")
            self._write_launch_summary(log_file, config_path)
            log_file.flush()

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            self.job = {
                "id": job_id,
                "mode": mode,
                "model_id": model_id,
                "branch": branch,
                "dataset_profile": dataset_profile,
                "pid": self.process.pid,
                "status": "running",
                "returncode": None,
                "started_at": started_at.isoformat(),
                "finished_at": None,
                "log_file": str(log_path),
                "config_file": str(config_path),
                "command": cmd,
                "params": dict(payload),
                "output_dir": str(output_dir),
            }
            create_run_record(self.project_root, {
                "id": job_id,
                "kind": "training",
                "status": "running",
                "mode": mode,
                "model_id": model_id,
                "branch": branch,
                "dataset_profile": dataset_profile or "default",
                "pid": self.process.pid,
                "started_at": started_at.isoformat(),
                "log_file": self.relative_path(log_path),
                "config_file": self.relative_path(config_path),
                "command": [str(part) for part in cmd],
                "params": dict(payload),
                "output_dir": self.relative_path(output_dir),
            })

            self.reader_thread = threading.Thread(
                target=self._capture_stdout,
                args=(self.process, log_file),
                daemon=True,
            )
            self.reader_thread.start()
            return self._status_locked()

    def _runtime_backend(self):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"
            xpu_backend = getattr(torch, "xpu", None)
            if xpu_backend and xpu_backend.is_available():
                return "xpu"
        except Exception as exc:
            logger.debug("Unable to detect training backend: %s", exc, exc_info=True)
        return "cpu"

    def _validate_branch_backend(self, branch):
        try:
            with open(self.base_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except OSError as exc:
            raise LocalTuneError("CONFIG_UNAVAILABLE", "Unable to read project configuration") from exc
        branch_config = ((config.get("quantization") or {}).get("branches") or {}).get(branch, {})
        if branch == "nvfp4" or branch_config.get("load_mode") == "nvfp4_qlora" or branch_config.get("quant_type") == "nvfp4":
            raise LocalTuneError(
                "BRANCH_UNSUPPORTED",
                "NVFP4 is not a supported training branch in LocalTune Console",
            )
        supported = branch_config.get("supported_backends")
        if not supported:
            load_mode = branch_config.get("load_mode", "")
            supported = ["cuda"] if load_mode in {"bnb_qlora", "unsloth_qlora"} else ["cuda"]
        backend = self._runtime_backend()
        if backend not in supported:
            raise LocalTuneError(
                "BRANCH_BACKEND_UNSUPPORTED",
                f"Load method '{branch}' supports {', '.join(supported)}, but current backend is {backend}",
            )

    def stop(self):
        with self.lock:
            self._refresh_locked()
            if not self.process or self.process.poll() is not None:
                return self._status_locked()
            self.job["status"] = "stopping"
            update_run_record(self.project_root, self.job["id"], {"status": "stopping"})
            self.process.terminate()
            return self._status_locked()

    def shutdown(self, timeout=10):
        """Stop an active child process and let the reader persist final state."""
        with self.lock:
            process = self.process
            reader_thread = self.reader_thread
            job = dict(self.job) if self.job else None
            if process and process.poll() is None and self.job:
                self.job["status"] = "stopping"
                update_run_record(self.project_root, self.job["id"], {"status": "stopping"})

        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=timeout)

        with self.lock:
            self._refresh_locked()
            if job and self.job and self.job.get("status") in {"running", "stopping"}:
                finished_at = datetime.now().isoformat()
                self.job.update(status="interrupted", finished_at=finished_at)
                update_run_record(
                    self.project_root,
                    self.job["id"],
                    {"status": "interrupted", "finished_at": finished_at},
                )
            return self._status_locked()

    def status(self):
        with self.lock:
            self._refresh_locked()
            if not self.job:
                return {"status": "idle", "job": None}
            return self._status_locked()

    def active_log_file(self):
        with self.lock:
            if self.job and self.job.get("log_file"):
                path = Path(self.job["log_file"])
                if path.exists():
                    return path
        return None

    def _capture_stdout(self, process, log_file):
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
        finally:
            rc = process.wait()
            log_file.write(f"\n[dashboard] process exited with code {rc}\n")
            log_file.flush()
            log_file.close()
            job_snapshot = None
            with self.lock:
                if self.process is process and self.job:
                    self.job["returncode"] = rc
                    self.job["finished_at"] = datetime.now().isoformat()
                    if self.job.get("status") == "stopping":
                        self.job["status"] = "stopped"
                    else:
                        self.job["status"] = "completed" if rc == 0 else "failed"
                    run_updates = {
                        "status": self.job["status"],
                        "returncode": rc,
                        "finished_at": self.job["finished_at"],
                    }
                    if rc != 0:
                        diagnostics = self._failure_diagnostics(Path(self.job["log_file"]))
                        if diagnostics:
                            self.job["diagnostics"] = diagnostics
                            run_updates["diagnostics"] = diagnostics
                    update_run_record(self.project_root, self.job["id"], run_updates)
                    job_snapshot = dict(self.job)
            if job_snapshot and job_snapshot.get("status") == "completed":
                manifests = self._write_artifact_manifests(job_snapshot)
                if manifests:
                    update_run_record(self.project_root, job_snapshot["id"], {"artifact_manifests": manifests})
                    with self.lock:
                        if self.job and self.job.get("id") == job_snapshot["id"]:
                            self.job["artifact_manifests"] = manifests

    def _refresh_locked(self):
        if self.process and self.job:
            rc = self.process.poll()
            if rc is not None and self.job["status"] in {"running", "stopping"}:
                self.job["returncode"] = rc
                self.job["finished_at"] = datetime.now().isoformat()
                if self.job["status"] == "stopping":
                    self.job["status"] = "stopped"
                else:
                    self.job["status"] = "completed" if rc == 0 else "failed"
                update_run_record(self.project_root, self.job["id"], {
                    "status": self.job["status"],
                    "returncode": rc,
                    "finished_at": self.job["finished_at"],
                })

    def _status_locked(self):
        if not self.job:
            return {"status": "idle", "job": None}
        return {"status": self.job["status"], "job": dict(self.job)}

    def _recover_interrupted_runs(self):
        finished_at = datetime.now().isoformat()
        for record in list_run_records(self.project_root):
            if record.get("kind") != "training" or record.get("status") not in {"running", "stopping"}:
                continue
            update_run_record(
                self.project_root,
                record["id"],
                {
                    "status": "interrupted",
                    "finished_at": finished_at,
                    "diagnostics": {
                        "code": "dashboard_restarted",
                        "title": "Dashboard restarted",
                        "summary": "The dashboard restarted before the final training state was recorded.",
                        "suggestions": ["Check the task log before starting a replacement run."],
                    },
                },
            )

    def _write_runtime_config(self, job_id, mode, payload):
        with open(self.base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        quant = config.setdefault("quantization", {})
        branch = payload.get("branch", "bnb4")
        quant["active_branch"] = branch
        if payload.get("no_fallback", True):
            quant["fallback_chain"] = [branch]

        training = config.setdefault("training", {})
        model = config.setdefault("model", {})
        lora = model.setdefault("lora", {})
        evaluation = config.setdefault("evaluation", {})
        data = config.setdefault("data", {})
        self._apply_model_selection(model, quant, branch, payload.get("model_id"))

        dataset_profile = payload.get("dataset_profile")
        profiles = data.get("profiles", {}) or {}
        if not dataset_profile:
            raise RuntimeError("请选择语料档案")
        if dataset_profile not in profiles:
            raise RuntimeError(f"未知语料档案: {dataset_profile}")
        profile = profiles[dataset_profile] or {}
        if not profile.get("train_file"):
            raise RuntimeError(f"语料档案缺少训练集: {dataset_profile}")
        data["profile_id"] = dataset_profile
        for key in ["train_file", "val_file", "test_file"]:
            if profile.get(key):
                data[key] = profile[key]
            else:
                data.pop(key, None)

        if mode == "smoke":
            training["max_steps"] = int(payload.get("max_steps") or 1)
            training["max_seq_length"] = int(payload.get("max_seq_length") or 512)
            training["logging_steps"] = 1
            training["save_steps"] = int(payload.get("save_steps") or training["max_steps"])
            evaluation["do_eval"] = False
        else:
            max_steps = int(payload.get("max_steps") or -1)
            training["max_steps"] = max_steps
            training["max_seq_length"] = int(payload.get("max_seq_length") or training.get("max_seq_length", 512))
            training["logging_steps"] = int(payload.get("logging_steps") or training.get("logging_steps", 10))
            training["save_steps"] = int(payload.get("save_steps") or training.get("save_steps", 500))
            evaluation["do_eval"] = bool(payload.get("do_eval", evaluation.get("do_eval", False)))

        if payload.get("lora_r"):
            lora["r"] = int(payload["lora_r"])
        if payload.get("gradient_accumulation_steps"):
            training["gradient_accumulation_steps"] = int(payload["gradient_accumulation_steps"])

        training["dataloader_num_workers"] = int(payload.get("dataloader_num_workers") or 0)
        training["dataloader_pin_memory"] = False
        training["load_best_model_at_end"] = False
        if payload.get("resume_from_checkpoint"):
            training["resume_from_checkpoint"] = payload["resume_from_checkpoint"]
        else:
            training.pop("resume_from_checkpoint", None)

        self.runtime_config_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.runtime_config_dir / f"dashboard_{job_id}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        self._cleanup_runtime_configs()
        return config_path

    def _output_dir_from_config(self, config_path, branch):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        training = config.get("training", {}) or {}
        output_root = self.resolve_project_path(training.get("output_dir", DEFAULT_OUTPUT_DIR))
        if not output_root:
            raise LocalTuneError("OUTPUT_DIR_REQUIRED", "Training output directory is not configured")
        return output_root / branch

    def _apply_model_selection(self, model, quant, branch, model_id):
        catalog = model.get("models") or model.get("catalog") or {}
        selected_model = model_id or model.get("active_model")
        if not catalog:
            if selected_model:
                model["active_model"] = selected_model
            return

        if not selected_model:
            selected_model = next(iter(catalog), "")
        if selected_model not in catalog:
            raise RuntimeError(f"Unknown model: {selected_model}")

        item = catalog[selected_model] or {}
        branch_paths = item.get("paths") or item.get("branch_paths") or item.get("model_paths") or {}
        model_path = branch_paths.get(branch) or item.get("path") or item.get("model_path")
        if not model_path:
            raise RuntimeError(f"Model {selected_model} has no path for branch {branch}")

        model["active_model"] = selected_model
        model["name"] = item.get("name") or selected_model
        branches = quant.setdefault("branches", {})
        branch_config = branches.setdefault(branch, {})
        branch_config["model_path"] = model_path

    def _cleanup_runtime_configs(self):
        if self.runtime_config_keep <= 0:
            return
        configs = sorted(
            self.runtime_config_dir.glob("dashboard_*.yaml"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in configs[self.runtime_config_keep:]:
            try:
                path.unlink()
            except OSError:
                continue

    def _write_launch_summary(self, log_file, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            training = config.get("training", {}) or {}
            data = config.get("data", {}) or {}
            model = config.get("model", {}) or {}
            lora = model.get("lora", {}) or {}
            quant = config.get("quantization", {}) or {}

            def dataset_summary(label, raw_path):
                path = self.resolve_project_path(raw_path)
                if not path or not path.exists():
                    return f"[dashboard] dataset.{label}=missing path={raw_path}"
                rows = 0
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as dataset_file:
                        rows = sum(1 for line in dataset_file if line.strip())
                except OSError:
                    rows = -1
                size_kb = path.stat().st_size / 1024
                row_text = "unknown" if rows < 0 else str(rows)
                return f"[dashboard] dataset.{label}=rows={row_text}, size={size_kb:.1f}KB, path={self.relative_path(path)}"

            log_file.write("[dashboard] ---- run summary ----\n")
            log_file.write(f"[dashboard] config={self.relative_path(config_path)}\n")
            log_file.write(f"[dashboard] model.active_model={model.get('active_model', '-')}\n")
            log_file.write(f"[dashboard] model.name={model.get('name', '-')}\n")
            log_file.write(f"[dashboard] quant.active_branch={quant.get('active_branch', '-')}\n")
            branch_config = (quant.get("branches", {}) or {}).get(quant.get("active_branch", ""), {}) or {}
            log_file.write(f"[dashboard] model.path={branch_config.get('model_path', '-')}\n")
            log_file.write(f"[dashboard] dataset.profile={data.get('profile_id', '-')}\n")
            log_file.write(dataset_summary("train", data.get("train_file")) + "\n")
            log_file.write(dataset_summary("val", data.get("val_file")) + "\n")
            log_file.write(dataset_summary("test", data.get("test_file")) + "\n")
            log_file.write(
                "[dashboard] training="
                f"max_steps={training.get('max_steps')}, "
                f"epochs={training.get('num_train_epochs')}, "
                f"seq={training.get('max_seq_length')}, "
                f"batch={training.get('per_device_train_batch_size')}, "
                f"grad_accum={training.get('gradient_accumulation_steps')}, "
                f"logging_steps={training.get('logging_steps')}, "
                f"save_steps={training.get('save_steps')}\n"
            )
            log_file.write(
                "[dashboard] lora="
                f"r={lora.get('r')}, alpha={lora.get('lora_alpha')}, "
                f"dropout={lora.get('lora_dropout')}, targets={','.join(lora.get('target_modules', []))}\n"
            )
            log_file.write(f"[dashboard] output_dir={training.get('output_dir', '-')}\n")
            log_file.write("[dashboard] ---------------------\n")
        except Exception as exc:
            log_file.write(f"[dashboard] failed_to_write_run_summary={exc}\n")

    def _write_artifact_manifests(self, job):
        try:
            config_path = self.resolve_project_path(job.get("config_file"))
            if not config_path or not config_path.exists():
                return []
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            training = config.get("training", {}) or {}
            quant = config.get("quantization", {}) or {}
            branch = job.get("branch") or quant.get("active_branch", "bnb4")
            output_root = self.resolve_project_path(training.get("output_dir", DEFAULT_OUTPUT_DIR))
            if not output_root:
                return []
            output_dir = output_root / branch
            if not output_dir.exists():
                return []
            try:
                started_ts = datetime.fromisoformat(job.get("started_at", "")).timestamp() - 5
            except (TypeError, ValueError):
                started_ts = None

            written = []
            candidates = []
            final_dir = output_dir / "final"
            if final_dir.exists():
                candidates.append(("final_adapter", final_dir))
            candidates.extend(
                ("checkpoint", path)
                for path in sorted(output_dir.glob("checkpoint-*"), key=lambda item: item.stat().st_mtime)
                if path.is_dir()
            )
            runs_dir = output_dir / "runs"
            if runs_dir.exists():
                candidates.extend(
                    ("metrics_run", path)
                    for path in sorted([item for item in runs_dir.iterdir() if item.is_dir()], key=lambda item: item.stat().st_mtime)
                )

            for artifact_type, artifact_path in candidates:
                updated_ts = self._artifact_content_mtime(artifact_path, artifact_type)
                if started_ts and updated_ts and updated_ts < started_ts:
                    continue
                manifest = self._build_artifact_manifest(job, config, output_dir, artifact_path, artifact_type)
                manifest_path = write_artifact_manifest(artifact_path, manifest)
                written.append({
                    "artifact_type": artifact_type,
                    "artifact_path": self.relative_path(artifact_path),
                    "manifest_path": self.relative_path(manifest_path),
                })
            return written
        except Exception as exc:
            logger.warning("Failed to write artifact manifests for job %s: %s", job.get("id"), exc, exc_info=True)
            return []

    def _artifact_content_mtime(self, artifact_path, artifact_type):
        try:
            if artifact_type == "metrics_run":
                files = list(artifact_path.rglob("metrics.jsonl"))
            else:
                files = [
                    path
                    for path in artifact_path.rglob("*")
                    if path.is_file() and path.name != "localtune_artifact.json"
                ]
            if not files:
                return artifact_path.stat().st_mtime
            return max(path.stat().st_mtime for path in files)
        except OSError:
            return None

    def _build_artifact_manifest(self, job, config, output_dir, artifact_path, artifact_type):
        data = config.get("data", {}) or {}
        training = config.get("training", {}) or {}
        model = config.get("model", {}) or {}
        lora = model.get("lora", {}) or {}
        quant = config.get("quantization", {}) or {}
        datasets = {
            "profile": job.get("dataset_profile") or "",
            "train_file": self._dataset_manifest_info(data.get("train_file")),
            "val_file": self._dataset_manifest_info(data.get("val_file")),
            "test_file": self._dataset_manifest_info(data.get("test_file")),
        }
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "artifact": {
                "type": artifact_type,
                "name": artifact_path.name,
                "path": self.relative_path(artifact_path),
                "branch": job.get("branch") or quant.get("active_branch", ""),
            },
            "run": {
                "id": job.get("id"),
                "kind": "training",
                "mode": job.get("mode"),
                "model_id": job.get("model_id") or model.get("active_model"),
                "status": job.get("status"),
                "returncode": job.get("returncode"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "dataset_profile": datasets["profile"],
            },
            "paths": {
                "artifact": self.relative_path(artifact_path),
                "output_dir": self.relative_path(output_dir),
                "runtime_config": self.relative_path(job.get("config_file")),
                "log_file": self.relative_path(job.get("log_file")),
            },
            "datasets": datasets,
            "training": self._selected_keys(training, [
                "max_steps",
                "num_train_epochs",
                "max_seq_length",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "logging_steps",
                "save_steps",
                "save_total_limit",
                "output_dir",
            ]),
            "lora": self._selected_keys(lora, ["r", "lora_alpha", "lora_dropout", "target_modules"]),
            "model": {
                "id": job.get("model_id") or model.get("active_model"),
                "name": model.get("name", ""),
            },
            "quantization": {
                "active_branch": quant.get("active_branch"),
                "fallback_chain": quant.get("fallback_chain", []),
            },
        }

    def _dataset_manifest_info(self, value):
        if not value:
            return None
        path = self.resolve_project_path(value)
        info = {"path": self.relative_path(path) if path else str(value), "exists": bool(path and path.exists())}
        if path and path.exists() and path.is_file():
            info["size_bytes"] = path.stat().st_size
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    info["rows"] = sum(1 for line in f if line.strip())
            except OSError:
                info["rows"] = None
        return info

    def _selected_keys(self, mapping, keys):
        return {key: mapping.get(key) for key in keys if key in mapping}

    def _failure_diagnostics(self, log_path):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")[-2_000_000:]
        except OSError:
            return None
        lowered = text.lower()
        oom_signatures = (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "cublas_status_alloc_failed",
            "not enough memory",
        )
        if any(signature in lowered for signature in oom_signatures):
            return {
                "code": "cuda_oom",
                "title": "GPU 显存不足",
                "summary": "训练进程因 CUDA 显存不足而退出，任务未自动修改参数或重试。",
                "suggestions": [
                    "降低最大序列长度，优先从 512 调整到 384 或 256。",
                    "确认单卡 batch size 为 1；需要更大有效 batch 时提高梯度累积步数。",
                    "降低 LoRA 秩，或关闭其它占用 GPU 的程序后重新运行。",
                    "保留当前日志和参数后手动重试，避免自动降级导致训练配置不可复现。",
                ],
            }
        return {
            "code": "process_failed",
            "title": "训练进程异常退出",
            "summary": "请查看任务日志末尾的错误堆栈确认原因。",
            "suggestions": ["先检查模型路径、语料格式和依赖环境，再使用相同参数重试。"],
        }
