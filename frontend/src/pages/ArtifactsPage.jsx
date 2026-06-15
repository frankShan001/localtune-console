import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, ArchiveRestore, CheckCircle2, Copy, FileText, HardDrive, Layers3, ListFilter, PackageCheck, Search, Square, Star, Trash2, WandSparkles, X } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatBytes, formatDate, formatNumber, shortPath } from "../api.js";
import { ConfirmDialog, EmptyState, OpenFolderButton, PageToolbar, Panel, PropertyList, SelectControl, TabBar } from "../components/ui.jsx";
import { TaskContextDrawer } from "../components/taskContext.jsx";
import { buildTaskList } from "./taskSupport.js";

function artifactLabel(type, t) {
  if (type === "final_adapter") return t("finalAdapter");
  if (type === "checkpoint") return t("checkpoint");
  if (type === "metrics_run") return t("metricsRun");
  if (type === "merged_model") return t("mergedModel");
  return type;
}

function artifactDescription(item, t) {
  if (!item) return "";
  if (item.type === "final_adapter") return t("finalAdapterHint");
  if (item.type === "checkpoint") return t("checkpointHint");
  if (item.type === "metrics_run") return t("metricsRunHint");
  if (item.type === "merged_model") return t("mergedModelHint");
  return t("artifactHint");
}

function datasetPath(dataset) {
  if (!dataset) return "";
  return typeof dataset === "string" ? dataset : dataset.path;
}

function datasetDetail(dataset, t) {
  if (!dataset) return "";
  if (typeof dataset === "string") return dataset;
  const parts = [dataset.path];
  if (dataset.rows != null) parts.push(`${dataset.rows} ${t("rowsUnit")}`);
  if (dataset.size_bytes != null) parts.push(formatBytes(dataset.size_bytes));
  if (dataset.exists === false) parts.push(t("missing"));
  return parts.filter(Boolean).join(" · ");
}

function artifactPriority(item) {
  let score = 0;
  if (item.type === "final_adapter") score += 100;
  if (item.type === "merged_model") score += 90;
  if (item.is_adapter) score += 40;
  if (item.best) score += 30;
  if (item.has_manifest) score += 20;
  if (!item.archived) score += 10;
  return score;
}

function sortArtifactsForPicker(items) {
  return [...items].sort((a, b) => {
    const priority = artifactPriority(b) - artifactPriority(a);
    if (priority !== 0) return priority;
    return String(b.updated || "").localeCompare(String(a.updated || ""));
  });
}

function manifestRows(manifest, t) {
  const training = manifest?.training || {};
  const lora = manifest?.lora || {};
  const datasets = manifest?.datasets || {};
  const quant = manifest?.quantization || {};
  const paths = manifest?.paths || {};
  return [
    [t("runId"), manifest?.run?.id],
    [t("runMode"), manifest?.run?.mode],
    [t("runStatus"), manifest?.run?.status ? t(`status_${manifest.run.status}`) : ""],
    [t("datasetProfile"), manifest?.run?.dataset_profile || datasets.profile],
    [t("config"), paths.runtime_config],
    [t("logFile"), paths.log_file],
    [t("activeBranch"), quant.active_branch],
    [t("maxSteps"), training.max_steps],
    [t("maxSeqLength"), training.max_seq_length],
    [t("learningRate"), training.learning_rate],
    [t("epochs"), training.num_train_epochs],
    [t("batchSize"), training.per_device_train_batch_size],
    [t("gradAccumulation"), training.gradient_accumulation_steps],
    [t("loraR"), lora.r],
    [t("loraAlpha"), lora.lora_alpha],
    [t("trainData"), datasetDetail(datasets.train_file, t)],
    [t("valData"), datasetDetail(datasets.val_file, t)],
    [t("testData"), datasetDetail(datasets.test_file, t)],
    [t("outputDir"), paths.output_dir],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function hashParam(name) {
  const query = String(window.location.hash || "").split("?")[1] || "";
  return new URLSearchParams(query).get(name) || "";
}

export function ArtifactsPage({ t }) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("all");
  const [selectedPath, setSelectedPath] = useState(() => hashParam("path"));
  const [selectedTaskFile, setSelectedTaskFile] = useState("");
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [artifactSearch, setArtifactSearch] = useState("");
  const [branchFilter, setBranchFilter] = useState("all");
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const [exportOutputPath, setExportOutputPath] = useState("");
  const [exportDtype, setExportDtype] = useState("bf16");
  const [exportTrustRemoteCode, setExportTrustRemoteCode] = useState(false);
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: () => apiGet("/api/artifacts"), refetchInterval: 8000 });
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => apiGet("/api/runs"), refetchInterval: 10000 });
  const logs = useQuery({ queryKey: ["log-history"], queryFn: () => apiGet("/api/logs/history"), refetchInterval: 10000 });
  const modelExports = useQuery({ queryKey: ["model-exports"], queryFn: () => apiGet("/api/model-exports"), refetchInterval: 5000 });

  const items = artifacts.data?.items || [];
  const activeItems = items.filter((item) => !item.archived);
  const counts = {
    all: activeItems.length,
    final_adapter: activeItems.filter((item) => item.type === "final_adapter").length,
    checkpoint: activeItems.filter((item) => item.type === "checkpoint").length,
    merged_model: activeItems.filter((item) => item.type === "merged_model").length,
    archived: items.filter((item) => item.archived).length,
  };
  const typeFilteredItems = filter === "archived"
    ? items.filter((item) => item.archived)
    : activeItems.filter((item) => filter === "all" || item.type === filter);
  const branches = [...new Set(typeFilteredItems.map((item) => item.branch).filter(Boolean))].sort();
  const artifactKeyword = artifactSearch.trim().toLowerCase();
  const filteredItems = typeFilteredItems.filter((item) => (
    (branchFilter === "all" || item.branch === branchFilter)
    && (!artifactKeyword || [item.name, item.path, item.run_id, item.dataset_profile]
      .some((value) => String(value || "").toLowerCase().includes(artifactKeyword)))
  ));
  const groupedItems = [
    {
      key: "final_adapter",
      label: t("deployableAdapters"),
      description: t("finalAdapterHint"),
      items: sortArtifactsForPicker(filteredItems.filter((item) => item.type === "final_adapter")),
    },
    {
      key: "checkpoint",
      label: t("trainingCheckpoints"),
      description: t("checkpointHint"),
      items: sortArtifactsForPicker(filteredItems.filter((item) => item.type === "checkpoint")),
    },
    {
      key: "merged_model",
      label: t("mergedModels"),
      description: t("mergedModelHint"),
      items: sortArtifactsForPicker(filteredItems.filter((item) => item.type === "merged_model")),
    },
  ].filter((group) => filter === "all" ? group.items.length : group.key === filter);
  const selectedArtifact = filteredItems.find((item) => item.path === selectedPath) || filteredItems[0];
  const manifest = selectedArtifact?.manifest || null;
  const manifestRunId = manifest?.run?.id || selectedArtifact?.run_id || "";
  const allTasks = useMemo(() => buildTaskList(runs.data?.runs || [], logs.data?.logs || []), [runs.data, logs.data]);
  const relatedTasks = useMemo(() => {
    if (manifestRunId) {
      return allTasks.filter((task) => task.id === manifestRunId);
    }
    return [];
  }, [allTasks, manifestRunId]);
  const selectedTask = relatedTasks.find((task) => task.file === selectedTaskFile) || relatedTasks[0];
  const selectedTaskSummaryRows = manifest ? manifestRows(manifest, t) : [];
  const exportJobs = modelExports.data?.items || [];
  const selectedExportJobs = exportJobs.filter((item) => (
    item.adapter_path === selectedArtifact?.path || item.output_path === selectedArtifact?.path
  ));
  const latestExportJob = selectedExportJobs[0];
  const metrics = useQuery({
    queryKey: ["artifact-metrics", manifestRunId],
    queryFn: () => apiGet(`/api/metrics?job_id=${encodeURIComponent(manifestRunId)}`),
    enabled: Boolean(manifestRunId),
    refetchInterval: 10000,
  });
  const relationLabel = selectedArtifact?.has_manifest ? t("exactRelation") : t("inferredRelation");
  const relationHint = selectedArtifact?.has_manifest ? t("hasManifest") : t("noManifest");
  const exactSourceRows = manifest ? [
    [t("runId"), manifest.run?.id],
    [t("datasetProfile"), manifest.run?.dataset_profile || manifest.datasets?.profile],
    [t("runMode"), manifest.run?.mode],
    [t("runFinishedAt"), formatDate(manifest.run?.finished_at)],
    [t("runtimeConfig"), manifest.paths?.runtime_config],
    [t("logFile"), manifest.paths?.log_file],
    [t("trainData"), datasetPath(manifest.datasets?.train_file)],
  ] : [];
  const copyInferenceArgs = () => {
    if (!selectedArtifact?.path) return;
    navigator.clipboard?.writeText(`--adapter "${selectedArtifact.path}" --branch "${selectedArtifact.branch || ""}"`);
  };
  const manage = useMutation({
    mutationFn: ({ action, path }) => apiPost("/api/artifacts/manage", { action, path }),
    onSuccess: (data) => {
      setSelectedPath(data.action === "delete" ? "" : data.path);
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const startExport = useMutation({
    mutationFn: () => apiPost("/api/model-exports/start", {
      adapter_path: selectedArtifact.path,
      output_path: exportOutputPath || undefined,
      dtype: exportDtype,
      trust_remote_code: exportTrustRemoteCode,
    }),
    onSuccess: () => {
      setExportOutputPath("");
      queryClient.invalidateQueries({ queryKey: ["model-exports"] });
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    },
  });
  const cancelExport = useMutation({
    mutationFn: (jobId) => apiPost(`/api/model-exports/${encodeURIComponent(jobId)}/cancel`, {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-exports"] }),
  });
  const openInference = () => {
    if (!selectedArtifact?.path) return;
    sessionStorage.setItem("localtune.inferenceTarget", JSON.stringify({
      adapter: selectedArtifact.path,
      branch: selectedArtifact.branch,
      model_id: selectedArtifact.manifest?.model?.id || "",
      mode: "compare",
    }));
    window.location.hash = "#/inference";
  };
  const requestDelete = () => {
    if (!selectedArtifact) return;
    setDeleteCandidate(selectedArtifact);
  };

  useEffect(() => {
    const onHashChange = () => {
      const artifactPath = hashParam("path");
      if (artifactPath) setSelectedPath(artifactPath);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (filteredItems.length && !filteredItems.some((item) => item.path === selectedPath)) {
      setSelectedPath(filteredItems[0].path);
    }
  }, [filteredItems, selectedPath]);

  useEffect(() => {
    if (relatedTasks.length && !relatedTasks.some((task) => task.file === selectedTaskFile)) {
      setSelectedTaskFile(relatedTasks[0].file);
    }
  }, [relatedTasks, selectedTaskFile]);

  useEffect(() => {
    if (!isPickerOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") setIsPickerOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isPickerOpen]);

  return (
    <div className="artifact-manager stack">
      <PageToolbar
        icon={Archive}
        title={t("artifacts")}
        subtitle={t("artifactsHint")}
        stats={[
          { icon: CheckCircle2, label: t("deployableAdapters"), value: counts.final_adapter },
          { icon: Layers3, label: t("trainingCheckpoints"), value: counts.checkpoint },
          { icon: ArchiveRestore, label: t("archivedArtifacts"), value: counts.archived },
        ]}
        onRefresh={() => queryClient.invalidateQueries()}
        refreshing={artifacts.isFetching || runs.isFetching || logs.isFetching}
        refreshLabel={t("refresh")}
      />
      <section className="artifact-toolbar">
        <div className="artifact-toolbar-main compact">
          <TabBar value={filter} onChange={setFilter} items={[
            ["all", `${t("allArtifacts")} ${counts.all}`, Archive],
            ["final_adapter", `${t("deployableAdapters")} ${counts.final_adapter}`, CheckCircle2],
            ["checkpoint", `${t("trainingCheckpoints")} ${counts.checkpoint}`, Layers3],
            ["merged_model", `${t("mergedModels")} ${counts.merged_model}`, PackageCheck],
            ["archived", `${t("archivedArtifacts")} ${counts.archived}`, ArchiveRestore],
          ]} />
          <button className="secondary-button" onClick={() => setIsPickerOpen(true)}>
            <ListFilter size={16} />
            {t("selectArtifact")}
          </button>
        </div>
      </section>

      {isPickerOpen && (
        <div className="artifact-picker-backdrop" onMouseDown={() => setIsPickerOpen(false)}>
          <aside className="artifact-picker" onMouseDown={(event) => event.stopPropagation()}>
            <header className="artifact-picker-head">
              <div>
                <h2>{t("artifactList")}</h2>
                <p>{filteredItems.length} {t("artifactCount")}</p>
              </div>
              <button className="secondary-button icon-only" title={t("close")} onClick={() => setIsPickerOpen(false)}>
                <X size={18} />
              </button>
            </header>
            <div className="artifact-picker-filters">
              <label className="compact-search">
                <Search size={15} />
                <input value={artifactSearch} placeholder={t("searchArtifacts")} onChange={(event) => setArtifactSearch(event.target.value)} />
              </label>
              <SelectControl
                value={branchFilter}
                onChange={setBranchFilter}
                ariaLabel={t("allBranches")}
                options={[["all", t("allBranches")], ...branches.map((branch) => [branch, branch])]}
              />
            </div>
            <div className="artifact-list">
              {groupedItems.map((group) => (
                <section className="artifact-group" key={group.key}>
                  <div className="artifact-group-head">
                    <strong>{group.label}</strong>
                    <span>{group.items.length}</span>
                  </div>
                  {group.items.map((item) => (
                    <button
                      key={item.path}
                      className={selectedArtifact?.path === item.path ? "artifact-row active" : "artifact-row"}
                      onClick={() => {
                        setSelectedPath(item.path);
                        setIsPickerOpen(false);
                      }}
                    >
                      <HardDrive size={16} />
                      <span>
                        <strong>{item.name}</strong>
                        <small>{item.branch || "-"} · {formatBytes(item.size_bytes)} · {formatDate(item.updated)}</small>
                      </span>
                    </button>
                  ))}
                </section>
              ))}
              {!filteredItems.length && <EmptyState text={t("noData")} />}
            </div>
          </aside>
        </div>
      )}

      <div className="artifact-layout">
        <section className="artifact-detail-column">
          {!selectedArtifact && <EmptyState text={t("noArtifactsYet")} />}
          {selectedArtifact && (
            <>
              {selectedArtifact.is_adapter && !selectedArtifact.archived && (
                <Panel
                  className="merge-export-panel"
                  title={t("mergePublish")}
                  subtitle={t("mergePublishHint")}
                  actions={latestExportJob?.status === "running" ? (
                    <button className="secondary-button" disabled={cancelExport.isPending} onClick={() => cancelExport.mutate(latestExportJob.id)}>
                      <Square size={14} /> {t("cancel")}
                    </button>
                  ) : null}
                >
                  <div className="merge-export-grid">
                    <div className="field">
                      <label>{t("mergeBaseModel")}</label>
                      <input value={selectedArtifact.adapter_check?.base_model || t("autoResolveBaseModel")} readOnly />
                    </div>
                    <div className="field">
                      <label>{t("mergeOutputDir")}</label>
                      <input value={exportOutputPath} placeholder={t("autoOutputDir")} onChange={(event) => setExportOutputPath(event.target.value)} />
                    </div>
                    <div className="field">
                      <label>{t("mergeDtype")}</label>
                      <SelectControl
                        value={exportDtype}
                        onChange={setExportDtype}
                        ariaLabel={t("mergeDtype")}
                        options={[
                          ["bf16", "bf16"],
                          ["fp16", "fp16"],
                          ["fp32", "fp32"],
                        ]}
                      />
                    </div>
                    <label className="merge-export-checkbox">
                      <input type="checkbox" checked={exportTrustRemoteCode} onChange={(event) => setExportTrustRemoteCode(event.target.checked)} />
                      <span>{t("trustRemoteCode")}</span>
                    </label>
                    <button className="primary-button" disabled={startExport.isPending || latestExportJob?.status === "running"} onClick={() => startExport.mutate()}>
                      <PackageCheck size={16} /> {t("startMergePublish")}
                    </button>
                  </div>
                  {startExport.error && <div className="error-text">{apiErrorMessage(startExport.error, t)}</div>}
                  {latestExportJob && (
                    <div className="merge-export-status">
                      <div>
                        <span className={`status-pill ${latestExportJob.status}`}>{t(`status_${latestExportJob.status}`)}</span>
                        <strong>{latestExportJob.output_path}</strong>
                      </div>
                      <PropertyList rows={[
                        [t("logFile"), latestExportJob.log_file],
                        [t("startedAt"), formatDate(latestExportJob.started_at)],
                        [t("finishedAt"), formatDate(latestExportJob.finished_at)],
                        [t("returnCode"), latestExportJob.returncode],
                      ]} />
                      {latestExportJob.log_tail && <pre className="merge-export-log">{latestExportJob.log_tail}</pre>}
                    </div>
                  )}
                </Panel>
              )}

              <Panel
                className="artifact-main-panel"
                title={selectedArtifact.name}
                subtitle={artifactDescription(selectedArtifact, t)}
                actions={(
                  <>
                    <button className="secondary-button" onClick={() => navigator.clipboard?.writeText(selectedArtifact.path)}>
                      <Copy size={15} /> {t("copyPath")}
                    </button>
                    {selectedArtifact.is_adapter && (
                      <button className="secondary-button" onClick={copyInferenceArgs}>
                        <Copy size={15} /> {t("copyInferenceArgs")}
                      </button>
                    )}
                    {selectedArtifact.is_adapter && !selectedArtifact.archived && (
                      <button className="secondary-button" onClick={openInference}>
                        <WandSparkles size={15} /> {t("validateArtifact")}
                      </button>
                    )}
                    {selectedArtifact.has_manifest && !selectedArtifact.archived && (
                      <button className="secondary-button" disabled={manage.isPending} onClick={() => manage.mutate({ action: selectedArtifact.best ? "unbest" : "best", path: selectedArtifact.path })}>
                        <Star size={15} /> {selectedArtifact.best ? t("unmarkBest") : t("markBest")}
                      </button>
                    )}
                    {selectedArtifact.has_manifest && (
                      <button className="secondary-button" disabled={manage.isPending} onClick={() => manage.mutate({ action: selectedArtifact.archived ? "restore" : "archive", path: selectedArtifact.path })}>
                        <ArchiveRestore size={15} /> {selectedArtifact.archived ? t("restoreArtifact") : t("archiveArtifact")}
                      </button>
                    )}
                    <button className="danger-button" disabled={manage.isPending} onClick={requestDelete}>
                      <Trash2 size={15} /> {t("delete")}
                    </button>
                    <button className="secondary-button" disabled={!selectedTask} onClick={() => setContextOpen(true)}>
                      <FileText size={15} /> {t("openTaskContext")}
                    </button>
                    <OpenFolderButton path={selectedArtifact.path} label={t("openFolder")} />
                  </>
                )}
              >
                <div className="artifact-detail-head">
                  <div className="artifact-type">{artifactLabel(selectedArtifact.type, t)}</div>
                  <div className={selectedArtifact.has_manifest ? "relation-badge exact" : "relation-badge inferred"}>{relationLabel}</div>
                  {selectedArtifact.best && <div className="relation-badge best">{t("bestArtifact")}</div>}
                  {selectedArtifact.archived && <div className="relation-badge archived">{t("archived")}</div>}
                </div>
                <div className="artifact-summary-list">
                  <PropertyList rows={[
                    [t("branch"), selectedArtifact.branch],
                    [t("path"), selectedArtifact.path],
                    [t("runId"), selectedArtifact.run_id],
                    [t("datasetProfile"), selectedArtifact.dataset_profile],
                    [t("size"), formatBytes(selectedArtifact.size_bytes)],
                    [t("updated"), formatDate(selectedArtifact.updated)],
                    [t("artifactUsability"), selectedArtifact.adapter_check ? (selectedArtifact.adapter_check.ok ? t("valid") : t("invalid")) : "-"],
                    [t("adapterBaseModel"), selectedArtifact.adapter_check?.base_model],
                    [t("loraR"), selectedArtifact.adapter_check?.r],
                    [t("loraAlpha"), selectedArtifact.adapter_check?.lora_alpha],
                    [t("sourceReliability"), relationHint],
                  ]} />
                </div>
                {selectedArtifact.adapter_check?.errors?.length > 0 && (
                  <div className="error-text">{selectedArtifact.adapter_check.errors.join(" · ")}</div>
                )}
                {manage.error && <div className="error-text">{apiErrorMessage(manage.error, t)}</div>}
              </Panel>

              <div className="two-column">
                <Panel title={t("artifactTrainingSource")} subtitle={manifest ? t("exactRelationHint") : t("weakRelationHint")}>
                  {manifest ? (
                    <PropertyList rows={exactSourceRows} />
                  ) : (
                    <EmptyState text={t("noSourceRecordDetail")} />
                  )}
                </Panel>

                <Panel title={t("metricsSummary")} subtitle={metrics.data?.source || ""}>
                  <PropertyList rows={[
                    [t("latestLoss"), formatNumber(metrics.data?.latest?.train_loss?.value, 4)],
                    [t("loss"), `${metrics.data?.train_loss?.length || 0} ${t("rowsUnit")}`],
                    [t("source"), metrics.data?.source],
                  ]} />
                </Panel>
              </div>

              {manifest && (
                <>
                  <Panel title={t("trainingParamSummary")} subtitle={t("exactRelationHint")}>
                    {selectedTaskSummaryRows.length ? (
                      <PropertyList rows={selectedTaskSummaryRows} />
                    ) : (
                      <EmptyState text={t("noData")} />
                    )}
                  </Panel>
                </>
              )}

            </>
          )}
        </section>
      </div>
      <TaskContextDrawer
        open={contextOpen}
        onClose={() => setContextOpen(false)}
        task={selectedTask}
        artifacts={items}
        t={t}
      />
      <ConfirmDialog
        open={Boolean(deleteCandidate)}
        title={t("deleteArtifactTitle")}
        message={deleteCandidate ? t("deleteArtifactConfirm", { name: deleteCandidate.name }) : ""}
        confirmLabel={t("confirmDelete")}
        cancelLabel={t("cancel")}
        pending={manage.isPending}
        onCancel={() => setDeleteCandidate(null)}
        onConfirm={() => {
          if (!deleteCandidate) return;
          manage.mutate({ action: "delete", path: deleteCandidate.path });
          setDeleteCandidate(null);
        }}
      />
    </div>
  );
}
