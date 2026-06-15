const now = "2026-06-12T10:00:00.000Z";

const modelSuitability = {
  available_vram_gb: 23.9,
  backend: "cuda",
  estimated_vram_gb: 22.2,
  load_method: "bnb4",
  params_b: 27,
  reason: "near_vram_limit",
  status: "caution",
};

const branch = {
  framework: "peft",
  id: "bnb4",
  path: "./models/Qwen/Qwen3.6-27B",
  path_exists: true,
  path_resolved: "models/Qwen/Qwen3.6-27B",
  quant_type: "nf4",
};

const model = {
  active: true,
  branches: [branch],
  description: "Validated local 27B base model for documentation screenshots",
  id: "qwen3_27b",
  name: "Qwen/Qwen3.6-27B",
  suitability: modelSuitability,
};

const splitFiles = {
  train: {
    exists: true,
    extension: ".jsonl",
    folder: "data/processed",
    id: "train:data/processed/editorial_rewrite_train.jsonl",
    inferred_format: "localtune_v1",
    name: "editorial_rewrite_train.jsonl",
    path: "data/processed/editorial_rewrite_train.jsonl",
    role: "train",
    rows: 480,
    size_bytes: 820000,
    task_type: "rewrite",
    trainable: true,
    updated: now,
  },
  val: {
    exists: true,
    extension: ".jsonl",
    folder: "data/processed",
    id: "val:data/processed/editorial_rewrite_val.jsonl",
    inferred_format: "localtune_v1",
    name: "editorial_rewrite_val.jsonl",
    path: "data/processed/editorial_rewrite_val.jsonl",
    role: "val",
    rows: 32,
    size_bytes: 56000,
    task_type: "rewrite",
    trainable: true,
    updated: now,
  },
  test: {
    exists: true,
    extension: ".jsonl",
    folder: "data/processed",
    id: "test:data/processed/editorial_rewrite_test.jsonl",
    inferred_format: "localtune_v1",
    name: "editorial_rewrite_test.jsonl",
    path: "data/processed/editorial_rewrite_test.jsonl",
    role: "test",
    rows: 32,
    size_bytes: 57000,
    task_type: "rewrite",
    trainable: true,
    updated: now,
  },
};

const datasetProfile = {
  description: "Editorial rewrite dataset for tone and structure adaptation.",
  files: splitFiles,
  format: "localtune_v1",
  id: "editorial_rewrite",
  name: "Editorial Rewrite Dataset",
  task_type: "rewrite",
  total_rows: 544,
  total_size_bytes: 933000,
  train: splitFiles.train,
  val: splitFiles.val,
  test: splitFiles.test,
  validation: {
    error_count: 0,
    ok: true,
    rows: 544,
    warning_count: 0,
  },
};

const trainingPayload = {
  branch: "bnb4",
  dataset_profile: "editorial_rewrite",
  gradient_accumulation_steps: 16,
  logging_steps: 1,
  lora_r: 16,
  max_seq_length: 512,
  max_steps: 1,
  mode: "smoke",
  model_id: "qwen3_27b",
  no_fallback: true,
  save_steps: 1,
};

function repairForDependency(id) {
  if (["torch", "bitsandbytes"].includes(id)) return { mode: "auto", hint: id === "torch" ? "repairTorchCuda" : "repairBitsAndBytesCuda" };
  if (["node", "npm", "transformers", "peft", "trl", "accelerate", "datasets"].includes(id)) return { mode: "launcher", hint: "repairByLauncher" };
  if (id === "cuda") return { mode: "manual", hint: "repairCudaManual" };
  if (id === "nvidia_driver") return { mode: "manual", hint: "repairDriverManual" };
  if (id === "compute_backend") return { mode: "manual", hint: "repairComputeManual" };
  if (id === "python") return { mode: "manual", hint: "repairPythonManual" };
  if (id === "unsloth") return { mode: "manual", hint: "repairOptionalManual" };
  return { mode: "manual", hint: "repairUnknownHint" };
}

const dependencies = [
  ["python", "Python", "3.12.11", ">=3.12", "ready", true, "Project Python runtime"],
  ["torch", "PyTorch", "2.9.1+cu130", "CUDA build", "ready", true, "CUDA-enabled training runtime"],
  ["transformers", "Transformers", "4.56.0", "installed", "ready", true, "Model loading and tokenization"],
  ["peft", "PEFT", "0.18.0", "installed", "ready", true, "LoRA adapter training"],
  ["trl", "TRL", "0.25.0", "installed", "ready", true, "SFT trainer"],
  ["accelerate", "Accelerate", "1.12.0", "installed", "ready", true, "Training runtime orchestration"],
  ["datasets", "Datasets", "4.5.0", "installed", "ready", true, "Dataset loading"],
  ["bitsandbytes", "bitsandbytes", "0.48.2", "CUDA", "ready", true, "NF4 quantized loading"],
  ["compute_backend", "Compute backend", "CUDA", "NVIDIA CUDA", "ready", true, "NVIDIA GeForce RTX 5090 Laptop GPU"],
  ["cuda", "CUDA", "13.0", "detected", "ready", true, "CUDA runtime detected"],
  ["nvidia_driver", "NVIDIA driver", "580.97", "detected", "ready", true, "Driver detected"],
  ["node", "Node.js", "24.16.0", ">=22", "ready", true, "Project-local frontend toolchain"],
  ["npm", "npm", "11.13.0", ">=10", "ready", true, "Frontend package manager"],
  ["unsloth", "Unsloth", "", "optional", "optional", false, "Optional integration"],
].map(([id, name, version, requirement, status, required, detail]) => ({
  id,
  name,
  version,
  requirement,
  status,
  required,
  detail,
  repair: repairForDependency(id),
}));

const artifacts = [
  {
    archived: false,
    best: true,
    branch: "bnb4",
    config_file: "configs/runtime/demo_editorial_rewrite.yaml",
    dataset_profile: "editorial_rewrite",
    exists: true,
    has_manifest: true,
    is_adapter: true,
    log_file: "logs/demo_editorial_rewrite.log",
    name: "final-adapter",
    path: "outputs/localtune/bnb4/final-adapter",
    run_id: "demo_run_001",
    size_bytes: 41000000,
    type: "final_adapter",
    updated: now,
  },
  {
    archived: false,
    best: false,
    branch: "bnb4",
    dataset_profile: "editorial_rewrite",
    exists: true,
    has_manifest: true,
    is_adapter: false,
    name: "checkpoint-20",
    path: "outputs/localtune/bnb4/checkpoint-20",
    run_id: "demo_run_001",
    size_bytes: 83000000,
    type: "checkpoint",
    updated: now,
  },
];

const runs = [
  {
    branch: "bnb4",
    dataset_profile: "editorial_rewrite",
    file: "logs/demo_editorial_rewrite.log",
    finished_at: now,
    id: "demo_run_001",
    kind: "training",
    mode: "smoke",
    model_id: "qwen3_27b",
    output_dir: "outputs/localtune/bnb4",
    params: trainingPayload,
    returncode: 0,
    started_at: "2026-06-12T09:58:00.000Z",
    status: "completed",
  },
];

const goldenStatus = {
  blockers: [],
  can_start_smoke: true,
  checked_at: now,
  metrics: {
    data_quality: "ready",
    evaluation_readiness: "ready",
    smoke_success: "done",
    time_to_evaluated_adapter: "not measured yet",
  },
  model_guidance: {
    available_vram_gb: 23.9,
    backend: "cuda",
    caution_max_params_b: 27,
    device_name: "NVIDIA GeForce RTX 5090 Laptop GPU",
    load_method: "bnb4",
    reason: "estimated_from_vram",
    recommended_max_params_b: 14,
    status: "ready",
  },
  next_step: {
    action: "Evaluate adapter",
    detail: "Compare Base vs Adapter and save a report",
    id: "evaluate",
    label: "Evaluate",
    route: "inference",
    status: "ready",
  },
  ok: true,
  payloads: {
    formal: { ...trainingPayload, mode: "formal", logging_steps: 10, save_steps: 500 },
    smoke: trainingPayload,
  },
  score: 89,
  selection: {
    adapter: artifacts[0],
    branch,
    dataset_profile: datasetProfile,
    model,
  },
  steps: [
    { action: "Fix environment", detail: "13/13 required dependencies ready, NVIDIA GeForce RTX 5090 Laptop GPU", id: "environment", label: "Environment", route: "environment", status: "done" },
    { action: "Select model", detail: "Qwen/Qwen3.6-27B", id: "model", label: "Model", route: "models", status: "done" },
    { action: "Import data", detail: "Editorial Rewrite Dataset · 544 rows", id: "dataset", label: "Dataset", route: "corpus", status: "done" },
    { action: "Start test run", detail: "Last test run: demo_run_001", id: "smoke", label: "Test Run", route: "golden", status: "done" },
    { action: "Start formal run", detail: "Adapter available: final-adapter", id: "train", label: "Train", route: "training", status: "done" },
    { action: "Evaluate adapter", detail: "Compare Base vs Adapter and save a report", id: "evaluate", label: "Evaluate", route: "inference", status: "ready" },
  ],
  training_readiness: { can_train: true, code: "ready", route: "golden" },
  training_status: { job: null, status: "idle" },
};

const modelRecommendations = {
  accelerator: {
    available: true,
    backend: "cuda",
    device_count: 1,
    device_name: modelSuitability.device_name || "NVIDIA GeForce RTX 5090 Laptop GPU",
    max_memory: modelSuitability.available_vram_gb,
    source: "torch-cuda",
  },
  downloads: [],
  locale: "zh",
  ok: true,
  provider: "modelscope",
  recommendations: [
    {
      download_available: true,
      download_command: "uv run --isolated --no-project --with modelscope python scripts/download_model.py --provider modelscope --model Qwen/Qwen3.5-9B --output models",
      download_url: "https://modelscope.cn/models/Qwen/Qwen3.5-9B",
      family: "Qwen",
      fit: { reason: "has_headroom", status: "recommended" },
      id: "qwen3_5_9b",
      language_fit: "zh_en",
      min_vram_gb: 12,
      name: "Qwen3.5 9B",
      params_b: 9,
      provider_model_id: "Qwen/Qwen3.5-9B",
      provider_name: "Qwen/Qwen3.5-9B",
      recommended_vram_gb: 16,
      summary: "????????? 2026 ????? Qwen ???",
    },
    {
      download_available: true,
      download_command: "uv run --isolated --no-project --with modelscope python scripts/download_model.py --provider modelscope --model Qwen/Qwen3.6-27B --output models",
      download_url: "https://modelscope.cn/models/Qwen/Qwen3.6-27B",
      family: "Qwen",
      fit: { reason: "near_limit", status: "caution" },
      id: "qwen3_6_27b",
      language_fit: "zh_en",
      min_vram_gb: 22,
      name: "Qwen3.6 27B",
      params_b: 27,
      provider_model_id: "Qwen/Qwen3.6-27B",
      provider_name: "Qwen/Qwen3.6-27B",
      recommended_vram_gb: 32,
      summary: "???????? Qwen3.6 ???????????????",
    },
  ],
  target_dir: "models",
};

const config = {
  active_branch: "bnb4",
  active_model: "qwen3_27b",
  branches: [{
    compatibility: { backend: "cuda", reason: "", supported: true },
    description: "bitsandbytes NF4 QLoRA with PEFT",
    framework: "peft",
    id: "bnb4",
    load_mode: "bnb_qlora",
    model_path: "./models/Qwen/Qwen3.6-27B",
    model_path_exists: true,
    model_path_resolved: "models/Qwen/Qwen3.6-27B",
    quant_type: "nf4",
    supported_backends: ["cuda"],
  }],
  config_file: "configs/model_config.yaml",
  dataset_format: "localtune_v1",
  model_scan_dirs: [{ exists: true, path: "models", path_resolved: "E:/localtune/models" }],
  models: [model],
  monitoring: { dashboard_port: 6543 },
  project: { description: "Local fine-tuning control center", name: "localtune-console", version: "0.4.0" },
  project_root: "E:/localtune",
  runtime_backend: {
    available: true,
    backend: "cuda",
    device_count: 1,
    device_name: "NVIDIA GeForce RTX 5090 Laptop GPU",
    max_memory: 23.9,
    memory_allocated: 0,
    memory_reserved: 0,
    memory_used: 0,
    source: "torch-cuda",
  },
  task_type: "rewrite",
  training: {
    gradient_accumulation_steps: 16,
    learning_rate: 0.0001,
    max_seq_length: 512,
    max_steps: 1,
    output_dir: "./outputs/localtune",
  },
};

const registry = {
  files: Object.values(splitFiles),
  format: "localtune_v1",
  materials: [
    {
      extension: ".txt",
      folder: "data/source_materials",
      id: "library:data/source_materials/brand_voice_notes.txt",
      inferred_format: "raw_text",
      name: "brand_voice_notes.txt",
      path: "data/source_materials/brand_voice_notes.txt",
      role: "library",
      rows: null,
      size_bytes: 24000,
      task_type: "raw_text",
      trainable: false,
      updated: now,
    },
  ],
  profiles: [datasetProfile],
  root: "data",
  schema_version: "localtune.dataset_registry.v1",
  summary: {
    file_count: 3,
    material_count: 1,
    material_size_bytes: 24000,
    profile_count: 1,
    total_file_count: 4,
    total_rows: 544,
    total_size_bytes: 957000,
  },
  task_type: "rewrite",
};

const preview = {
  errors: [],
  info: splitFiles.train,
  pagination: { has_next: true, has_previous: false, limit: 20, offset: 0, total: 480 },
  query: "",
  samples: [
    {
      line: 1,
      normalized: {
        format: "localtune_v1",
        input: "The product update was released on Friday with several improvements.",
        instruction: "Rewrite this update in a warm editorial voice.",
        metadata: { source: "demo", split: "train", style: "editorial" },
        output: "On Friday, the team shipped a cleaner, steadier product update with improvements designed to make daily work feel easier.",
        system: "You are an editorial rewrite assistant. Preserve facts and improve tone.",
        task_type: "rewrite",
        trainable: true,
      },
      row: {},
    },
    {
      line: 2,
      normalized: {
        format: "localtune_v1",
        input: "The report summarizes customer feedback from the last release.",
        instruction: "Rewrite this note for an internal product newsletter.",
        metadata: { source: "demo", split: "train", style: "newsletter" },
        output: "This report gathers what customers told us after the last release and turns those signals into a clearer product story.",
        system: "You are an editorial rewrite assistant. Preserve facts and improve tone.",
        task_type: "rewrite",
        trainable: true,
      },
      row: {},
    },
  ],
};

const environmentDependencies = {
  accelerator: config.runtime_backend,
  checked_at: now,
  counts: { required_ready: 14, required_total: 14 },
  items: dependencies,
  platform: { machine: "x86_64", system: "Windows" },
};

const status = {
  cpu: { percent: 12 },
  gpu: { available: true, memory_used: 2.1, memory_total: 23.9, memory_percent: 8.8, utilization: 16, name: "NVIDIA GeForce RTX 5090 Laptop GPU", temperature: 48, power_draw: 28.5 },
  memory: { percent: 41, used_gb: 12.9, total_gb: 31.4 },
};

const metrics = {
  latest_loss: 5.7631,
  points: [
    { step: 1, loss: 5.91 },
    { step: 10, loss: 5.82 },
    { step: 20, loss: 5.76 },
  ],
  source: "outputs/localtune/bnb4/runs/demo_run_001",
};

const logs = {
  file: "logs/demo_editorial_rewrite.log",
  lines: [
    "[localtune] Preparing model Qwen/Qwen3.6-27B with bnb4/NF4 QLoRA",
    "[localtune] Loaded Editorial Rewrite Dataset: train=480 val=32 test=32",
    "[localtune] step=1 loss=5.91",
    "[localtune] step=10 loss=5.82",
    "[localtune] step=20 loss=5.76",
    "[localtune] Saved final adapter to outputs/localtune/bnb4/final-adapter",
  ],
  path: "logs/demo_editorial_rewrite.log",
  text: "",
};
logs.text = logs.lines.join("\n");

const demoRoutes = {
  "/api/artifacts": { artifacts, output_root: "outputs/localtune" },
  "/api/config": config,
  "/api/corpus/library": registry,
  "/api/corpus/preview": preview,
  "/api/datasets": { profiles: [datasetProfile] },
  "/api/datasets/registry": registry,
  "/api/environment/dependencies": environmentDependencies,
  "/api/golden-path/status": goldenStatus,
  "/api/logs": logs,
  "/api/logs/history": { files: [{ file: logs.file, kind: "training", modified: now, size_bytes: logs.text.length }] },
  "/api/metrics": metrics,
  "/api/models/recommendations": modelRecommendations,
  "/api/recipes": { recipes: [] },
  "/api/runs": { runs },
  "/api/status": status,
  "/api/training/status": { job: null, status: "idle" },
};

export function isDocsDemoMode() {
  return new URLSearchParams(window.location.search).get("demo") === "docs";
}

export function demoGet(path) {
  if (!isDocsDemoMode()) return null;
  const url = new URL(path, window.location.origin);
  const value = demoRoutes[url.pathname];
  return value ? structuredClone(value) : null;
}
