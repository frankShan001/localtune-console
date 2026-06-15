export function durationText(started, finished, elapsedSeconds) {
  const seconds = Number(elapsedSeconds);
  if (Number.isFinite(seconds) && seconds > 0) return `${Math.round(seconds)} s`;
  if (!started || !finished) return "-";
  const start = new Date(started).getTime();
  const end = new Date(finished).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return "-";
  const diff = Math.round((end - start) / 1000);
  if (diff < 60) return `${diff} s`;
  const minutes = Math.floor(diff / 60);
  const rest = diff % 60;
  return `${minutes} min ${rest} s`;
}

export function normalizeRun(run) {
  return {
    id: run.id,
    source: "run",
    file: run.log_file,
    kind: "training",
    mode: run.mode || "training",
    status: run.status || "unknown",
    branch: run.branch || "",
    datasetProfile: run.dataset_profile || "",
    updated: run.updated_at || run.finished_at || run.started_at || run.created_at,
    started: run.started_at || run.created_at,
    finished: run.finished_at,
    outputDir: run.output_dir || "",
    params: run.params || {},
    command: run.command || [],
    returncode: run.returncode,
    elapsedSeconds: run.elapsed_seconds,
    summary: run.summary || {},
    raw: run,
  };
}

export function normalizeLegacyLog(item) {
  return {
    id: item.job_id || item.name || item.file,
    source: "log",
    file: item.file,
    kind: "training",
    mode: item.mode || "training",
    status: item.status || "unknown",
    branch: item.branch || "",
    datasetProfile: item.dataset_profile || "",
    updated: item.updated,
    started: "",
    finished: item.updated,
    outputDir: "",
    params: {},
    command: [],
    returncode: undefined,
    elapsedSeconds: undefined,
    summary: item.summary || {},
    raw: item,
  };
}

export function buildTaskList(runs = [], logs = []) {
  const runTasks = runs
    .filter((run) => !run.kind || run.kind === "training")
    .map(normalizeRun);
  const runLogFiles = new Set(runTasks.map((task) => task.file).filter(Boolean));
  const legacyTasks = logs
    .filter((item) => item.file && item.mode !== "inference" && !runLogFiles.has(item.file))
    .map(normalizeLegacyLog);
  return [...runTasks, ...legacyTasks].sort((a, b) => String(b.updated || "").localeCompare(String(a.updated || "")));
}

export function taskKindLabel(task, t) {
  if (!task) return t("noData");
  return t("trainingTasks");
}

export function taskTitle(task, t) {
  if (!task) return t("noData");
  const kind = taskKindLabel(task, t);
  const mode = task.mode && task.mode !== task.kind ? t(task.mode) : kind;
  return `${mode} · ${task.branch || "-"}`;
}

export function artifactMatchesTaskSource(item, task) {
  if (!item || !task) return false;
  const itemRunId = item.run_id || item.manifest?.run?.id || "";
  const itemLogFile = item.log_file || item.manifest?.paths?.log_file || "";
  const taskFile = task.file || task.log_file || task.logFile || "";
  const normalizedItemLog = normalizeComparablePath(itemLogFile);
  const normalizedTaskLog = normalizeComparablePath(taskFile);
  return Boolean(
    (itemRunId && itemRunId === task.id)
    || (normalizedItemLog && normalizedTaskLog && normalizedItemLog === normalizedTaskLog)
    || (normalizedItemLog && normalizedTaskLog && normalizedTaskLog.endsWith(normalizedItemLog))
  );
}

export function artifactHasExactTaskSource(item, task) {
  if (!item || !task?.id) return false;
  const itemRunId = item.run_id || item.manifest?.run?.id || "";
  return Boolean(itemRunId && itemRunId === task.id);
}

export function buildTaskRerunPayload(task) {
  if (!task || task.source !== "run") return null;
  const params = task.params || {};
  const payload = {
    ...params,
    mode: params.mode || task.mode || "smoke",
    model_id: params.model_id || task.raw?.model_id || "",
    branch: params.branch || task.branch || "bnb4",
    dataset_profile: params.dataset_profile || task.datasetProfile || "",
  };
  if (!payload.model_id || !payload.dataset_profile) return null;
  return payload;
}

export function normalizeComparablePath(value) {
  return String(value || "").replaceAll("\\", "/").replace(/^[A-Za-z]:/, "").replace(/^\/+/, "");
}
