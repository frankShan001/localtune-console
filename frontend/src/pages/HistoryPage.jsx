import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, BookMarked, CheckCircle2, Clock3, Copy, FileText, RotateCcw, Search, XCircle } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatBytes, formatDate, formatNumber, shortPath } from "../api.js";
import { LogViewer } from "../components/monitoring.jsx";
import { TaskContextDrawer } from "../components/taskContext.jsx";
import { EmptyState, OpenFolderButton, PageToolbar, Panel, PropertyList, SelectControl } from "../components/ui.jsx";
import { artifactHasExactTaskSource, buildTaskList, buildTaskRerunPayload, durationText, taskTitle } from "./taskSupport.js";

function statusIcon(status) {
  if (status === "completed") return CheckCircle2;
  if (status === "failed") return XCircle;
  return Clock3;
}

function artifactLabel(type, t) {
  if (type === "final_adapter") return t("finalAdapter");
  if (type === "checkpoint") return t("checkpoint");
  if (type === "metrics_run") return t("metricsRun");
  return type;
}

function taskParamRows(task, t) {
  const params = task?.params || {};
  return [
    [t("baseModel"), params.model_id || task?.raw?.model_id],
    [t("datasetProfile"), params.dataset_profile || task?.datasetProfile],
    [t("branch"), params.branch || task?.branch],
    [t("maxSteps"), params.max_steps],
    [t("maxSeqLength"), params.max_seq_length],
    [t("loraR"), params.lora_r],
    [t("gradAccumulation"), params.gradient_accumulation_steps],
    [t("loggingSteps"), params.logging_steps],
    [t("saveSteps"), params.save_steps],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function hashParam(name) {
  const query = String(window.location.hash || "").split("?")[1] || "";
  return new URLSearchParams(query).get(name) || "";
}

export function HistoryPage({ t }) {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState(() => hashParam("task"));
  const [contextOpen, setContextOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortOrder, setSortOrder] = useState("newest");
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => apiGet("/api/runs"), refetchInterval: 8000 });
  const history = useQuery({ queryKey: ["log-history"], queryFn: () => apiGet("/api/logs/history"), refetchInterval: 8000 });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: () => apiGet("/api/artifacts"), refetchInterval: 12000 });
  const trainingStatus = useQuery({ queryKey: ["training-status"], queryFn: () => apiGet("/api/training/status"), refetchInterval: 5000 });

  const allTasks = useMemo(() => {
    return buildTaskList(runs.data?.runs || [], history.data?.logs || []);
  }, [runs.data, history.data]);
  const tasks = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    const filtered = allTasks.filter((task) => {
      if (statusFilter !== "all" && task.status !== statusFilter) return false;
      if (!keyword) return true;
      return [
        task.id,
        task.mode,
        task.branch,
        task.datasetProfile,
        task.raw?.model_id,
        task.file,
      ].some((value) => String(value || "").toLowerCase().includes(keyword));
    });
    return filtered.sort((a, b) => {
      const result = String(b.updated || "").localeCompare(String(a.updated || ""));
      return sortOrder === "oldest" ? -result : result;
    });
  }, [allTasks, search, sortOrder, statusFilter]);

  const selectedTask = tasks.find((task) => task.id === selectedId) || tasks[0];
  const relatedArtifacts = (artifacts.data?.items || []).filter((item) => artifactHasExactTaskSource(item, selectedTask));
  const rerunPayload = buildTaskRerunPayload(selectedTask);
  const statusRunning = trainingStatus.data?.status === "running" || trainingStatus.data?.status === "stopping";
  const rerun = useMutation({
    mutationFn: (payload) => apiPost("/api/training/start", payload),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const exportRecipe = useMutation({
    mutationFn: (runId) => apiPost("/api/recipes/export", { run_id: runId }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["recipes"] }),
  });
  const taskMetrics = useQuery({
    queryKey: ["task-metrics", selectedTask?.id],
    queryFn: () => apiGet(`/api/metrics?job_id=${encodeURIComponent(selectedTask.id)}`),
    enabled: Boolean(selectedTask?.id && selectedTask.source === "run"),
    refetchInterval: 12000,
  });
  const log = useQuery({
    queryKey: ["task-log", selectedTask?.file],
    queryFn: () => apiGet(`/api/logs?n=1800&file=${encodeURIComponent(selectedTask.file)}`),
    enabled: Boolean(selectedTask?.file),
  });

  useEffect(() => {
    const onHashChange = () => {
      const taskId = hashParam("task");
      if (taskId) setSelectedId(taskId);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (tasks.length && !tasks.some((task) => task.id === selectedId)) {
      setSelectedId(tasks[0].id);
    }
  }, [tasks, selectedId]);

  const completed = allTasks.filter((task) => task.status === "completed").length;
  const failed = allTasks.filter((task) => task.status === "failed").length;
  const running = allTasks.filter((task) => ["running", "stopping"].includes(task.status)).length;
  const StatusIcon = statusIcon(selectedTask?.status);
  const refreshing = runs.isFetching || history.isFetching || artifacts.isFetching || trainingStatus.isFetching;

  return (
    <div className="task-center stack">
      <PageToolbar
        icon={Clock3}
        title={t("taskCenter")}
        subtitle={t("taskCenterHint")}
        stats={[
          { icon: FileText, label: t("trainingTasks"), value: allTasks.length },
          { icon: Clock3, label: t("running"), value: running },
          { icon: CheckCircle2, label: t("completed"), value: completed },
          { icon: XCircle, label: t("failed"), value: failed },
        ]}
        onRefresh={() => queryClient.invalidateQueries()}
        refreshing={refreshing}
        refreshLabel={t("refresh")}
      />

      <section className="task-filter-bar">
        <label className="task-search">
          <Search size={16} />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t("searchTasks")} />
        </label>
        <SelectControl
          value={statusFilter}
          onChange={setStatusFilter}
          ariaLabel={t("allStatuses")}
          options={[
            ["all", t("allStatuses")],
            ["running", t("status_running")],
            ["completed", t("status_completed")],
            ["failed", t("status_failed")],
            ["stopped", t("status_stopped")],
          ]}
        />
        <SelectControl
          value={sortOrder}
          onChange={setSortOrder}
          ariaLabel={t("newestFirst")}
          options={[
            ["newest", t("newestFirst")],
            ["oldest", t("oldestFirst")],
          ]}
        />
        <span>{tasks.length} {t("filteredTasks")}</span>
      </section>

      <div className="task-layout">
        <aside className="task-list-column">
          <Panel title={t("taskList")} subtitle={`${tasks.length} ${t("trainingTasks")}`}>
            <div className="task-list">
              {tasks.map((task) => {
                const Icon = statusIcon(task.status);
                return (
                  <button key={task.id} className={selectedTask?.id === task.id ? "task-row active" : "task-row"} onClick={() => setSelectedId(task.id)}>
                    <Icon size={17} />
                    <span>
                      <span className="task-row-head">
                        <strong>{taskTitle(task, t)}</strong>
                        <b className={`task-row-state ${task.status}`}>{t(`status_${task.status}`)}</b>
                      </span>
                      <small className="task-row-meta">
                        <span>{task.datasetProfile || task.id} · {task.raw?.model_id || "-"}</span>
                        <em>{formatDate(task.updated)}</em>
                      </small>
                    </span>
                  </button>
                );
              })}
              {!tasks.length && <EmptyState text={t("noData")} />}
            </div>
          </Panel>
        </aside>

        <section className="task-detail-column">
          {!selectedTask && <EmptyState text={t("noTaskSelected")} />}
          {selectedTask && (
            <>
              <Panel
                title={t("taskDetail")}
                subtitle={selectedTask.id}
                actions={(
                  <>
                    <div className={`status-pill ${selectedTask.status}`}>
                      <StatusIcon size={16} /> {t(`status_${selectedTask.status}`)}
                    </div>
                    <button className="secondary-button" onClick={() => navigator.clipboard?.writeText(selectedTask.id)}>
                      <Copy size={15} /> {t("copyId")}
                    </button>
                    <button className="secondary-button" onClick={() => setContextOpen(true)}>
                      <FileText size={15} /> {t("openTaskContext")}
                    </button>
                    <button className="secondary-button" disabled={!rerunPayload || statusRunning || rerun.isPending} onClick={() => rerunPayload && rerun.mutate(rerunPayload)}>
                      <RotateCcw size={15} /> {t("rerunTask")}
                    </button>
                    <button className="secondary-button" disabled={selectedTask.source !== "run" || exportRecipe.isPending} onClick={() => exportRecipe.mutate(selectedTask.id)}>
                      <BookMarked size={15} /> {t("exportRecipe")}
                    </button>
                    {exportRecipe.data?.path && <OpenFolderButton path={exportRecipe.data.path} label={t("openRecipe")} />}
                    <OpenFolderButton path={selectedTask.outputDir || selectedTask.file} label={t("openFolder")} />
                  </>
                )}
              >
                <div className="task-detail-summary">
                  <div>
                    <span>{t("runKind")}</span>
                    <strong>{t("trainingTasks")}</strong>
                  </div>
                  <div>
                    <span>{t("mode")}</span>
                    <strong>{selectedTask.mode ? t(selectedTask.mode) : "-"}</strong>
                  </div>
                  <div>
                    <span>{t("branch")}</span>
                    <strong>{selectedTask.branch || "-"}</strong>
                  </div>
                  <div>
                    <span>{t("dataset")}</span>
                    <strong>{selectedTask.datasetProfile || "-"}</strong>
                  </div>
                  <div>
                    <span>{t("elapsed")}</span>
                    <strong>{durationText(selectedTask.started, selectedTask.finished, selectedTask.elapsedSeconds)}</strong>
                  </div>
                  <div>
                    <span>{t("returnCode")}</span>
                    <strong>{selectedTask.returncode ?? "-"}</strong>
                  </div>
                </div>
                <div className="task-detail-paths">
                  <div><span>{t("timeRange")}</span><strong>{formatDate(selectedTask.started)} → {formatDate(selectedTask.finished)}</strong></div>
                  <div><span>{t("outputDir")}</span><strong title={selectedTask.outputDir}>{selectedTask.outputDir || "-"}</strong></div>
                  <div><span>{t("logFile")}</span><strong title={selectedTask.file}>{selectedTask.file || "-"}</strong></div>
                </div>
                {selectedTask.source === "log" && <p className="muted-note">{t("legacyTaskHint")}</p>}
              </Panel>

              <div className="task-insight-grid">
                <Panel title={t("taskParams")}>
                  {taskParamRows(selectedTask, t).length ? (
                    <PropertyList rows={taskParamRows(selectedTask, t)} />
                  ) : (
                    <EmptyState text={t("noData")} />
                  )}
                </Panel>

                <div className="task-result-stack">
                  <Panel title={t("metricsSummary")}>
                    <div className="task-metric-strip">
                      <div><span>{t("latestLoss")}</span><strong>{formatNumber(taskMetrics.data?.latest?.train_loss?.value, 4)}</strong></div>
                      <div><span>{t("metricPoints")}</span><strong>{taskMetrics.data?.train_loss?.length || 0}</strong></div>
                      <div><span>{t("runStatus")}</span><strong>{t(`status_${selectedTask.status}`)}</strong></div>
                    </div>
                    <div className="task-metric-source">
                      <span>{t("source")}</span>
                      <strong title={taskMetrics.data?.source || ""}>{taskMetrics.data?.source || "-"}</strong>
                    </div>
                  </Panel>

                  <Panel title={t("relatedArtifacts")} subtitle={selectedTask.branch || ""}>
                    <div className="related-artifacts compact">
                      {relatedArtifacts.map((item) => (
                        <article className="related-artifact" key={item.path}>
                          <Archive size={16} />
                          <span>
                            <strong>{artifactLabel(item.type, t)}</strong>
                            <small title={item.path}>{shortPath(item.path)} · {formatBytes(item.size_bytes)}</small>
                          </span>
                          <OpenFolderButton path={item.path} label={t("openFolder")} />
                        </article>
                      ))}
                      {!relatedArtifacts.length && <EmptyState text={t("noData")} />}
                    </div>
                  </Panel>
                </div>
              </div>

              {(rerun.error || exportRecipe.error) && (
                <div className="error-text">{apiErrorMessage(rerun.error || exportRecipe.error, t)}</div>
              )}

              <Panel title={t("taskLogs")} subtitle={log.data?.file || selectedTask.file || ""}>
                <LogViewer lines={log.data?.logs || []} t={t} />
              </Panel>
            </>
          )}
        </section>
      </div>
      <TaskContextDrawer
        open={contextOpen}
        onClose={() => setContextOpen(false)}
        task={selectedTask}
        artifacts={artifacts.data?.items || []}
        t={t}
      />
    </div>
  );
}
