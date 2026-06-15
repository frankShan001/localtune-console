import React, { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Archive, BookMarked, Copy, Database, Gauge, Loader2, Play, Rocket, RotateCcw, ShieldCheck, SlidersHorizontal, Square, WandSparkles } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatBytes, shortPath } from "../api.js";
import { paramHelp } from "../config/appConfig.jsx";
import { EmptyState, InputField, OpenFolderButton, PageToolbar, Panel, PropertyList, SelectControl, SelectField, TabBar } from "../components/ui.jsx";
import { LogViewer, LossChart, ResourceBars, TaskStatus } from "../components/monitoring.jsx";
import { DatasetProfileSummary } from "./corpusSupport.jsx";
import { coerceTrainingPayload } from "./trainingSupport.js";
import { artifactHasExactTaskSource, buildTaskRerunPayload } from "./taskSupport.js";

function profileFileName(profile, role) {
  return shortPath(profile?.[role]?.path) || "-";
}

function profileOptionLabel(profile) {
  const name = profile.name || profile.id;
  return `${name} · train: ${profileFileName(profile, "train")} · val: ${profileFileName(profile, "val")}`;
}

function modelOptionLabel(model) {
  return model.name || model.id;
}

function modelBranchInfo(model, branchId) {
  return (model?.branches || []).find((branch) => branch.id === branchId);
}

function branchOptionLabel(branch, t) {
  const supported = branch.supported_backends?.length ? branch.supported_backends.join(", ") : "-";
  return `${branch.id} · ${t("supportedComputeBackends")}: ${supported}`;
}

function branchCompatibilityError(branch, t) {
  if (!branch?.compatibility || branch.compatibility.supported) return "";
  return `${t("backendUnsupported")} · ${t("currentComputeBackend")}: ${branch.compatibility.backend || "-"}`;
}

function LaunchSection({ icon: Icon, title, children }) {
  return (
    <section className="launch-section">
      <header>
        <Icon size={16} />
        <h3>{title}</h3>
      </header>
      {children}
    </section>
  );
}

const TRAINING_SCHEME_PRESETS = {
  smoke: { max_steps: 20, logging_steps: 1, save_steps: 20 },
  full: { max_steps: "", logging_steps: "", save_steps: "" },
};

const TRAINING_SCHEMES = [
  {
    id: "smoke",
    icon: Play,
    titleKey: "smokePlan",
    hintKey: "smokePlanHint",
    presetKey: "smokePlanPreset",
  },
  {
    id: "full",
    icon: Rocket,
    titleKey: "fullPlan",
    hintKey: "fullPlanHint",
    presetKey: "fullPlanPreset",
  },
];

function profileTrainRows(profile) {
  return Number(profile?.train?.rows ?? profile?.train?.line_count ?? profile?.train?.count ?? profile?.rows ?? 0) || 0;
}

function findPreferredProfile(mode, profiles) {
  const readyProfiles = (profiles || []).filter((profile) => profile?.train?.exists && profile.validation?.ok !== false);
  if (!readyProfiles.length) return null;
  const ranked = [...readyProfiles].sort((a, b) => profileTrainRows(a) - profileTrainRows(b));
  if (mode === "smoke") {
    return ranked.find((profile) => /mini|sample|small|smoke|test/i.test(`${profile.id} ${profile.name || ""}`))
      || ranked.find((profile) => profileTrainRows(profile) > 0 && profileTrainRows(profile) <= 50)
      || ranked[0];
  }
  return ranked.find((profile) => /full|formal|complete|train/i.test(`${profile.id} ${profile.name || ""}`))
    || ranked[ranked.length - 1];
}

function TrainingSchemePicker({ value, onChange, t }) {
  return (
    <div className="training-scheme-grid" role="radiogroup" aria-label={t("trainingPlan")}>
      {TRAINING_SCHEMES.map((scheme) => {
        const Icon = scheme.icon;
        const selected = value === scheme.id;
        return (
          <button
            type="button"
            role="radio"
            aria-checked={selected}
            className={selected ? "training-scheme-card selected" : "training-scheme-card"}
            key={scheme.id}
            onClick={() => onChange(scheme.id)}
          >
            <span className="training-scheme-icon"><Icon size={18} /></span>
            <span className="training-scheme-copy">
              <strong>{t(scheme.titleKey)}</strong>
              <small>{t(scheme.hintKey)}</small>
            </span>
            <em>{t(scheme.presetKey)}</em>
          </button>
        );
      })}
    </div>
  );
}

function artifactLabel(type, t) {
  if (type === "final_adapter") return t("finalAdapter");
  if (type === "checkpoint") return t("checkpoint");
  if (type === "metrics_run") return t("metricsRun");
  return type;
}

function traceFromStatusJob(job) {
  if (!job) return null;
  return {
    id: job.id,
    status: job.status,
    mode: job.mode,
    modelId: job.model_id,
    branch: job.branch,
    datasetProfile: job.dataset_profile,
    started: job.started_at,
    finished: job.finished_at,
    logFile: job.log_file,
    configFile: job.config_file,
    outputDir: job.output_dir,
    params: job.params || {},
    source: "run",
    raw: job,
  };
}

function formFromJob(job, current) {
  const params = job?.params || {};
  return {
    ...current,
    mode: params.mode || job?.mode || current.mode,
    model_id: params.model_id || job?.model_id || current.model_id,
    branch: params.branch || job?.branch || current.branch,
    dataset_profile: params.dataset_profile || job?.dataset_profile || current.dataset_profile,
    max_steps: params.max_steps ?? current.max_steps,
    max_seq_length: params.max_seq_length ?? current.max_seq_length,
    lora_r: params.lora_r ?? current.lora_r,
    gradient_accumulation_steps: params.gradient_accumulation_steps ?? current.gradient_accumulation_steps,
    logging_steps: params.logging_steps ?? current.logging_steps,
    save_steps: params.save_steps ?? current.save_steps,
    resume_from_checkpoint: params.resume_from_checkpoint || "",
    no_fallback: params.no_fallback ?? current.no_fallback,
  };
}

function isFinishedTrace(trace) {
  return ["completed", "failed", "stopped"].includes(trace?.status);
}

function goTo(page) {
  window.location.hash = `#/${page}`;
}

function TrainingTracePanel({ t, trace, artifacts, onRerun, rerunDisabled }) {
  const relatedArtifacts = artifacts
    .filter((item) => artifactHasExactTaskSource(item, trace))
    .sort((a, b) => String(b.updated || "").localeCompare(String(a.updated || "")));
  const finalAdapter = relatedArtifacts.find((item) => item.type === "final_adapter" && item.is_adapter);

  function validateFinalAdapter() {
    if (!finalAdapter) return;
    sessionStorage.setItem("localtune.inferenceTarget", JSON.stringify({
      model_id: finalAdapter.manifest?.model?.id || trace?.modelId || "",
      branch: finalAdapter.branch || trace?.branch || "bnb4",
      adapter: finalAdapter.path,
      mode: "compare",
    }));
    goTo("inference");
  }

  return (
    <Panel
      title={t("trainingTrace")}
      subtitle={trace?.id || t("waitingForTrainingTask")}
      actions={trace && (
        <>
          <button className="secondary-button" onClick={() => navigator.clipboard?.writeText(trace.id)}>
            <Copy size={15} /> {t("copyId")}
          </button>
          <OpenFolderButton path={trace.outputDir || trace.logFile} label={t("openFolder")} />
        </>
      )}
    >
      {!trace && (
        <div className="trace-empty">
          <strong>{t("noTrainingTrace")}</strong>
          <p>{t("trainingTraceEmptyHint")}</p>
          <ul>
            <li>{t("trainingTraceShowsConfig")}</li>
            <li>{t("trainingTraceShowsOutputs")}</li>
          </ul>
        </div>
      )}
      {trace && (
        <div className="training-trace">
          <PropertyList rows={[
            [t("baseModel"), trace.modelId || "-"],
            [t("branch"), trace.branch],
            [t("dataset"), trace.datasetProfile],
            [t("runtimeConfig"), trace.configFile],
            [t("outputDir"), trace.outputDir],
          ]} />

          <section className="trace-artifacts">
            <header>
              <Archive size={16} />
              <h3>{t("recordedArtifacts")}</h3>
            </header>
            <div className="related-artifacts compact">
              {relatedArtifacts.map((item) => (
                <article className="related-artifact" key={item.path}>
                  <Archive size={16} />
                  <span>
                    <strong>{artifactLabel(item.type, t)}</strong>
                    <small>{shortPath(item.path)} · {formatBytes(item.size_bytes)}</small>
                  </span>
                  <OpenFolderButton path={item.path} label={t("openFolder")} />
                </article>
              ))}
              {!relatedArtifacts.length && <EmptyState text={t("noRecordedArtifactsYet")} />}
            </div>
          </section>

          {isFinishedTrace(trace) && (
            <section className="post-run-actions">
              <div>
                <strong>{t("postTrainingActions")}</strong>
                <small>{t("postTrainingActionsHint")}</small>
              </div>
              <button className="secondary-button" onClick={() => goTo("artifacts")}>
                <Archive size={15} /> {t("viewArtifacts")}
              </button>
              <button className="secondary-button" disabled={!finalAdapter} onClick={validateFinalAdapter}>
                <WandSparkles size={15} /> {t("validateFinalAdapter")}
              </button>
              <button className="secondary-button" disabled={rerunDisabled} onClick={onRerun}>
                <RotateCcw size={15} /> {t("rerunTask")}
              </button>
            </section>
          )}
        </div>
      )}
    </Panel>
  );
}

export function TrainingPage({ t }) {
  const queryClient = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => apiGet("/api/datasets") });
  const config = useQuery({ queryKey: ["config"], queryFn: () => apiGet("/api/config") });
  const status = useQuery({ queryKey: ["training-status"], queryFn: () => apiGet("/api/training/status"), refetchInterval: 3000 });
  const runtime = useQuery({ queryKey: ["status"], queryFn: () => apiGet("/api/status"), refetchInterval: 4000 });
  const hasTrainingJob = Boolean(status.data?.job?.id);
  const liveJobId = status.data?.job?.id || "";
  const metrics = useQuery({ queryKey: ["metrics", liveJobId], queryFn: () => apiGet(`/api/metrics?job_id=${encodeURIComponent(liveJobId)}`), refetchInterval: 5000, enabled: hasTrainingJob });
  const logs = useQuery({ queryKey: ["logs-live", liveJobId], queryFn: () => apiGet("/api/logs?n=1000&kind=training"), refetchInterval: 3000, enabled: hasTrainingJob });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: () => apiGet("/api/artifacts"), refetchInterval: 8000 });
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: () => apiGet("/api/recipes") });
  const [selectedRecipe, setSelectedRecipe] = useState("");
  const [activeTab, setActiveTab] = useState("setup");
  const [form, setForm] = useState({
    mode: "smoke",
    model_id: "",
    branch: "bnb4",
    dataset_profile: "",
    max_steps: 20,
    max_seq_length: 512,
    lora_r: "",
    gradient_accumulation_steps: "",
    logging_steps: 1,
    save_steps: 20,
    resume_from_checkpoint: "",
    no_fallback: true,
  });

  useEffect(() => {
    if (config.data) {
      setForm((current) => {
        const profileExists = (datasets.data?.profiles || []).some((profile) => profile.id === current.dataset_profile);
        return {
          ...current,
          branch: config.data.active_branch || current.branch,
          dataset_profile: current.dataset_profile && profileExists ? current.dataset_profile : "",
          max_seq_length: config.data.training?.max_seq_length || current.max_seq_length,
          gradient_accumulation_steps: config.data.training?.gradient_accumulation_steps || "",
        };
      });
    }
  }, [config.data, datasets.data]);

  useEffect(() => {
    if (status.data?.job?.id) {
      setForm((current) => formFromJob(status.data.job, current));
      setActiveTab("monitor");
    }
  }, [status.data?.job?.id]);

  const start = useMutation({
    mutationFn: (payload) => apiPost("/api/training/start", coerceTrainingPayload(payload || form)),
    onSuccess: (data) => {
      if (data?.job?.id) {
        setForm((current) => formFromJob(data.job, current));
        setActiveTab("monitor");
      }
      queryClient.invalidateQueries();
    },
  });
  const stop = useMutation({
    mutationFn: () => apiPost("/api/training/stop"),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const applyRecipe = useMutation({
    mutationFn: () => apiPost("/api/recipes/import", { path: selectedRecipe }),
    onSuccess: (data) => setForm((current) => ({ ...current, ...data.payload, resume_from_checkpoint: "" })),
  });
  const running = status.data?.status === "running" || status.data?.status === "stopping";
  const models = config.data?.models || [];
  const selectedModel = models.find((model) => model.id === form.model_id);
  const selectedModelBranch = modelBranchInfo(selectedModel, form.branch);
  const selectedConfigBranch = (config.data?.branches || []).find((branch) => branch.id === form.branch);
  const backendError = branchCompatibilityError(selectedConfigBranch, t);
  const profiles = datasets.data?.profiles || [];
  const checkpoints = (artifacts.data?.items || []).filter((item) => item.type === "checkpoint");
  const selectedProfile = profiles.find((profile) => profile.id === form.dataset_profile);
  const readyProfileCount = profiles.filter((profile) => profile?.train?.exists && profile.validation?.ok !== false).length;
  const selectedScheme = TRAINING_SCHEMES.find((scheme) => scheme.id === form.mode) || TRAINING_SCHEMES[0];
  const trialRunLocksPreset = selectedScheme.id === "smoke";
  function applyTrainingScheme(mode) {
    const preferredProfile = findPreferredProfile(mode, profiles);
    setForm((current) => ({
      ...current,
      mode,
      ...TRAINING_SCHEME_PRESETS[mode],
      dataset_profile: preferredProfile?.id || current.dataset_profile,
    }));
  }
  const launchIssues = [
    !selectedModel ? t("needSelectModel") : "",
    selectedModel && !selectedModelBranch?.path_exists ? t("needValidModelPath") : "",
    backendError,
    !selectedProfile ? t("needSelectDataset") : "",
    selectedProfile && selectedProfile.validation?.ok === false ? t("needValidDataset") : "",
    !config.data?.training?.output_dir ? t("needOutputDir") : "",
  ].filter(Boolean);
  const startDisabledReason = running ? t("trainingAlreadyRunning") : launchIssues.join("；");
  const startDisabled = running || start.isPending || launchIssues.length > 0;
  const primaryLaunchIssue = launchIssues[0] || "";
  const trace = hasTrainingJob ? traceFromStatusJob(status.data?.job) : null;
  const rerunPayload = buildTaskRerunPayload(trace);
  const liveLossData = hasTrainingJob ? metrics.data?.train_loss || [] : [];
  const liveLogLines = hasTrainingJob ? logs.data?.logs || [] : [];
  const liveLogFile = hasTrainingJob ? logs.data?.file || "" : "";
  const modelError = !selectedModel ? t("needSelectModel") : "";
  const branchError = selectedModel && !selectedModelBranch?.path_exists ? t("needValidModelPath") : backendError;
  const datasetError = !selectedProfile ? t("needSelectDataset") : selectedProfile.validation?.ok === false ? t("needValidDataset") : "";
  const branchBackendHint = selectedConfigBranch
    ? `${t("currentComputeBackend")}: ${selectedConfigBranch.compatibility?.backend || config.data?.runtime_backend?.backend || "-"} · ${t("supportedComputeBackends")}: ${(selectedConfigBranch.supported_backends || []).join(", ") || "-"}`
    : "";
  const modelPathHint = [selectedModelBranch?.path ? `${t("modelPath")}: ${selectedModelBranch.path}` : "", branchBackendHint].filter(Boolean).join(" · ");
  const refreshing = [datasets, config, status, runtime, artifacts, recipes].some((query) => query.isFetching);

  return (
    <div className="training-page stack">
      <PageToolbar
        icon={Rocket}
        title={t("training")}
        subtitle={t("trainingHint")}
        stats={[
          { icon: Gauge, label: t("status"), value: t(`status_${status.data?.status || "idle"}`) },
          { icon: Database, label: t("registeredModels"), value: models.length },
          { icon: Database, label: t("readyTrainingDatasets"), value: `${readyProfileCount}/${profiles.length}` },
          { icon: Archive, label: t("trainingCheckpoints"), value: checkpoints.length },
        ]}
        onRefresh={() => queryClient.invalidateQueries()}
        refreshing={refreshing}
        refreshLabel={t("refresh")}
      />
      <div className="training-workspace">
        <TabBar value={activeTab} onChange={setActiveTab} items={[
          ["setup", t("trainingLaunch"), SlidersHorizontal],
          ["monitor", t("trainingMonitoring"), Gauge],
        ]} />
      {activeTab === "setup" && (
      <section className="training-inputs">
        <Panel
          title={t("trainingLaunch")}
          subtitle={t("trainingHint")}
          actions={(
            <>
              <button className="primary-button" title={startDisabledReason} disabled={startDisabled} onClick={() => start.mutate()}>
                {start.isPending ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
                {t("start")}
              </button>
              <button className="danger-button" disabled={!running || stop.isPending} onClick={() => stop.mutate()}>
                <Square size={15} /> {t("stop")}
              </button>
            </>
          )}
        >
          <div className="dataset-inventory-note">
            <span><strong>{readyProfileCount}</strong> / {profiles.length} {t("readyTrainingDatasets")}</span>
            <small>{profiles.length ? t("trainingDatasetInventoryHint") : t("trainingDatasetInventoryEmpty")}</small>
          </div>
          <div className="recipe-apply-bar">
            <BookMarked size={16} />
            <SelectControl
              value={selectedRecipe}
              onChange={setSelectedRecipe}
              ariaLabel={t("selectRecipe")}
              options={[
                ["", t("selectRecipe")],
                ...(recipes.data?.recipes || []).map((recipe) => [recipe.path, `${recipe.name} · ${recipe.model_id} · ${recipe.dataset_profile}`]),
              ]}
            />
            <button className="secondary-button" disabled={!selectedRecipe || applyRecipe.isPending} onClick={() => applyRecipe.mutate()}>
              {applyRecipe.isPending ? <Loader2 className="spin" size={16} /> : <BookMarked size={16} />}
              {t("applyRecipe")}
            </button>
          {applyRecipe.data?.recipe?.description && <span>{applyRecipe.data.recipe.description}</span>}
          </div>

          <div className="training-plan-block">
            <header>
              <div>
                <h3>{t("trainingPlan")}</h3>
                <p>{t("trainingPlanHint")}</p>
              </div>
              <span>{t(selectedScheme.id === "smoke" ? "smokeDatasetPolicy" : "fullDatasetPolicy")}</span>
            </header>
            <TrainingSchemePicker value={form.mode} onChange={applyTrainingScheme} t={t} />
          </div>

          <div className="launch-grid">
            <LaunchSection icon={Database} title={t("basicSetup")}>
              {primaryLaunchIssue && (
                <div className="launch-blocker">
                  <AlertTriangle size={15} />
                  <span>{primaryLaunchIssue}</span>
                </div>
              )}
              <div className="form-grid basic-training-grid">
                <SelectField label={t("baseModel")} help={paramHelp.model_id} value={form.model_id} onChange={(model_id) => setForm({ ...form, model_id })} options={[["", t("selectModel")], ...models.map((model) => [model.id, modelOptionLabel(model)])]} error={modelError} />
                <SelectField label={t("dataset")} help={paramHelp.dataset_profile} value={form.dataset_profile} onChange={(dataset_profile) => setForm({ ...form, dataset_profile })} options={[["", t("selectDatasetProfile")], ...profiles.map((profile) => [profile.id, profileOptionLabel(profile)])]} error={datasetError} />
                <SelectField label={t("branch")} help={paramHelp.branch} value={form.branch} onChange={(branch) => setForm({ ...form, branch })} options={(config.data?.branches || []).map((branch) => [branch.id, branchOptionLabel(branch, t)])} error={branchError} hint={modelPathHint} />
                <SelectField label={t("resumeFromCheckpoint")} help={paramHelp.resume_from_checkpoint} value={form.resume_from_checkpoint} onChange={(resume_from_checkpoint) => setForm({ ...form, resume_from_checkpoint })} options={[
                  ["", t("startFromBaseModel")],
                  ...checkpoints.map((item) => [item.path, `${t("continueFromCheckpoint")} · ${item.name} · ${item.run_id || t("inferredRelation")}`]),
                ]} />
              </div>
            </LaunchSection>

            <LaunchSection icon={Gauge} title={t("trainingScale")}>
              <div className="form-grid">
                <InputField label={t("maxSteps")} help={paramHelp.max_steps} value={form.max_steps} onChange={(max_steps) => setForm({ ...form, max_steps })} disabled={trialRunLocksPreset} hint={trialRunLocksPreset ? t("smokePlanPreset") : ""} />
                <InputField label={t("maxSeqLength")} help={paramHelp.max_seq_length} value={form.max_seq_length} onChange={(max_seq_length) => setForm({ ...form, max_seq_length })} />
                <InputField label={t("gradAccumulation")} help={paramHelp.gradient_accumulation_steps} value={form.gradient_accumulation_steps} onChange={(gradient_accumulation_steps) => setForm({ ...form, gradient_accumulation_steps })} placeholder={t("useConfig")} />
              </div>
            </LaunchSection>

            <LaunchSection icon={SlidersHorizontal} title={t("loraAndSaving")}>
              <div className="form-grid">
                <InputField label={t("loraR")} help={paramHelp.lora_r} value={form.lora_r} onChange={(lora_r) => setForm({ ...form, lora_r })} placeholder={t("useDefaultValue")} />
                <InputField label={t("loggingSteps")} help={paramHelp.logging_steps} value={form.logging_steps} onChange={(logging_steps) => setForm({ ...form, logging_steps })} disabled={trialRunLocksPreset} />
                <InputField label={t("saveSteps")} help={paramHelp.save_steps} value={form.save_steps} onChange={(save_steps) => setForm({ ...form, save_steps })} disabled={trialRunLocksPreset} />
              </div>
            </LaunchSection>

            <LaunchSection icon={ShieldCheck} title={t("safetyOptions")}>
              <label className="checkbox-row compact">
                <input type="checkbox" checked={form.no_fallback} onChange={(event) => setForm({ ...form, no_fallback: event.target.checked })} />
                {t("disableFallbackChain")}
              </label>
              <p className="model-selection-note">{t("loadingStrategyHint")}</p>
            </LaunchSection>
          </div>

          {selectedProfile && (
            <div className="profile-card compact training-profile-summary">
              <header>
                <div>
                  <h3>{t("selectedDatasetProfile")}</h3>
                  <p>{selectedProfile.name || form.dataset_profile}</p>
                </div>
                <span className={selectedProfile.validation?.ok ? "status-pill completed" : "status-pill failed"}>
                  {selectedProfile.validation?.ok ? t("valid") : t("invalid")}
                </span>
              </header>
              <DatasetProfileSummary t={t} profile={selectedProfile} />
            </div>
          )}
          {(start.error || stop.error || applyRecipe.error) && (
            <div className="error-text">{apiErrorMessage(start.error || stop.error || applyRecipe.error, t)}</div>
          )}
        </Panel>
      </section>
      )}
      {activeTab === "monitor" && (
      <section className="training-monitoring">
        <header className="training-monitoring-head">
          <h2>{t("trainingMonitoring")}</h2>
          <p>{t("trainingMonitoringHint")}</p>
        </header>
        <div className="training-monitor-grid">
          <Panel title={t("status")}>
            <TaskStatus data={status.data} progress={metrics.data?.progress} t={t} />
          </Panel>
          <TrainingTracePanel
            t={t}
            trace={trace}
            artifacts={artifacts.data?.items || []}
            onRerun={() => rerunPayload && start.mutate(rerunPayload)}
            rerunDisabled={running || start.isPending || !rerunPayload}
          />
          <Panel title={t("loss")}>
            <LossChart data={liveLossData} t={t} />
          </Panel>
          <Panel title={t("monitor")}>
            <ResourceBars status={runtime.data} t={t} />
          </Panel>
          <Panel title={t("taskLogs")} subtitle={liveLogFile}>
            {hasTrainingJob ? <LogViewer lines={liveLogLines} t={t} /> : <EmptyState text={t("noActiveTrainingLogs")} />}
          </Panel>
        </div>
      </section>
      )}
      </div>
    </div>
  );
}
