import React, { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Activity,
  ArrowRight,
  Box,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  Gauge,
  Map as MapIcon,
  Play,
  XCircle,
} from "lucide-react";
import { apiGet, formatDate, formatNumber } from "../api.js";
import { EmptyState, PageToolbar, Panel } from "../components/ui.jsx";
import { buildTaskList, durationText, normalizeRun, taskTitle } from "./taskSupport.js";

function goTo(route, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== ""),
  ).toString();
  window.location.hash = `#/${route}${query ? `?${query}` : ""}`;
}

function statusIcon(status) {
  if (status === "completed") return CheckCircle2;
  if (status === "failed") return XCircle;
  return Clock3;
}

function statusLabel(task, t) {
  return task?.status ? t(`status_${task.status}`) : t("status_idle");
}

function OverviewMetric({ icon: Icon, label, value, detail, tone = "neutral", onClick }) {
  return (
    <button className={`overview-metric ${tone}`} onClick={onClick}>
      <span className="overview-metric-icon"><Icon size={18} /></span>
      <span className="overview-metric-copy">
        <small>{label}</small>
        <strong>{value}</strong>
        <em>{detail}</em>
      </span>
      <ArrowRight size={16} />
    </button>
  );
}

function QuickStartStep({ step, t }) {
  const status = step?.status || "blocked";
  const Icon = status === "done" ? CheckCircle2 : status === "ready" ? Play : Clock3;
  return (
    <button className={`overview-quick-step ${status}`} onClick={() => goTo(step?.route || "golden")}>
      <span><Icon size={15} /></span>
      <strong>{t(`goldenStep_${step?.id || "golden"}`)}</strong>
      <small>{t(`goldenStatus_${status}`)}</small>
    </button>
  );
}

function TaskRow({ task, artifacts, t }) {
  const Icon = statusIcon(task.status);
  const artifactCount = artifacts.filter((item) => item.run_id && item.run_id === task.id).length;
  const steps = task.params?.max_steps ?? task.summary?.training?.max_steps;
  return (
    <button className="overview-task-row" onClick={() => goTo("history", { task: task.id })}>
      <span className={`overview-row-status ${task.status}`}><Icon size={16} /></span>
      <span className="overview-row-main">
        <strong>{taskTitle(task, t)}</strong>
        <small>{task.datasetProfile || t("notSelected")} · {steps ? `${steps} steps` : t("noData")}</small>
      </span>
      <span className="overview-row-meta">
        <strong>{statusLabel(task, t)}</strong>
        <small>{formatDate(task.updated)}</small>
      </span>
      <span className="overview-row-count">{artifactCount} {t("artifactCount")}</span>
      <ArrowRight size={15} />
    </button>
  );
}

function ArtifactRow({ item, t }) {
  const typeLabel = item.type === "final_adapter" ? t("finalAdapter") : t("checkpoint");
  return (
    <button className="overview-artifact-row" onClick={() => goTo("artifacts", { path: item.path })}>
      <span className="overview-artifact-icon"><Box size={16} /></span>
      <span>
        <strong>{item.name}</strong>
        <small>{typeLabel} · {item.branch || "-"}</small>
      </span>
      <span>
        <strong>{item.has_manifest ? t("exactRelation") : t("sourceMissing")}</strong>
        <small>{formatDate(item.updated)}</small>
      </span>
      <ArrowRight size={15} />
    </button>
  );
}

export function OverviewPage({ t }) {
  const queryClient = useQueryClient();
  const status = useQuery({ queryKey: ["training-status"], queryFn: () => apiGet("/api/training/status"), refetchInterval: 4000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => apiGet("/api/datasets") });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: () => apiGet("/api/artifacts"), refetchInterval: 10000 });
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => apiGet("/api/runs"), refetchInterval: 8000 });
  const history = useQuery({ queryKey: ["log-history"], queryFn: () => apiGet("/api/logs/history"), refetchInterval: 8000 });
  const config = useQuery({ queryKey: ["config"], queryFn: () => apiGet("/api/config") });
  const quickStart = useQuery({ queryKey: ["golden-path-status"], queryFn: () => apiGet("/api/golden-path/status"), refetchInterval: 6000 });

  const profiles = datasets.data?.profiles || [];
  const artifactItems = artifacts.data?.items || [];
  const tasks = useMemo(
    () => buildTaskList(runs.data?.runs || [], history.data?.logs || []),
    [history.data, runs.data],
  );
  const liveJob = status.data?.job;
  const focusTask = liveJob ? normalizeRun(liveJob) : null;
  const metrics = useQuery({
    queryKey: ["overview-metrics", focusTask?.id],
    queryFn: () => apiGet(`/api/metrics?job_id=${encodeURIComponent(focusTask.id)}`),
    enabled: Boolean(liveJob && focusTask?.id && focusTask.source === "run"),
    refetchInterval: liveJob ? 4000 : false,
  });

  const readyProfiles = profiles.filter((profile) => profile.validation?.ok && profile.train?.exists).length;
  const hasProfiles = profiles.length > 0;
  const models = config.data?.models || [];
  const usableModels = models.filter((model) => (model.branches || []).some((branch) => branch.path_exists)).length;
  const adapters = artifactItems.filter((item) => !item.archived && item.type === "final_adapter" && item.adapter_check?.ok);
  const recentArtifacts = artifactItems
    .filter((item) => !item.archived && ["final_adapter", "checkpoint"].includes(item.type))
    .sort((a, b) => String(b.updated || "").localeCompare(String(a.updated || "")))
    .slice(0, 4);
  const recentTasks = tasks.slice(0, 5);
  const failedRecentTasks = recentTasks.filter((task) => task.status === "failed");
  const invalidProfiles = profiles.filter((profile) => !profile.validation?.ok || !profile.train?.exists);
  const invalidAdapters = artifactItems.filter((item) => item.type === "final_adapter" && item.adapter_check && !item.adapter_check.ok);
  const unlinkedArtifacts = artifactItems.filter((item) => (
    !item.archived
    && ["final_adapter", "checkpoint"].includes(item.type)
    && !item.has_manifest
  ));
  const issues = [
    !hasProfiles && {
      text: t("noDatasetProfilesIssue"),
      route: "corpus",
    },
    hasProfiles && invalidProfiles.length && {
      text: t("invalidProfilesIssue", { count: invalidProfiles.length }),
      route: "corpus",
    },
    !usableModels && {
      text: t("noUsableModelsIssue"),
      route: "models",
    },
    invalidAdapters.length && {
      text: t("invalidAdaptersIssue", { count: invalidAdapters.length }),
      route: "artifacts",
    },
    failedRecentTasks.length && {
      text: t("failedTasksIssue", { count: failedRecentTasks.length }),
      route: "history",
    },
    unlinkedArtifacts.length && {
      text: t("unlinkedArtifactsIssue", { count: unlinkedArtifacts.length }),
      route: "artifacts",
    },
  ].filter(Boolean);

  const lossPoints = metrics.data?.train_loss || [];
  const firstLoss = Number(lossPoints[0]?.value);
  const latestLoss = Number(lossPoints.at(-1)?.value);
  const lossChange = Number.isFinite(firstLoss) && Number.isFinite(latestLoss) && firstLoss !== 0
    ? ((latestLoss - firstLoss) / firstLoss) * 100
    : null;
  const taskArtifacts = focusTask
    ? artifactItems.filter((item) => item.run_id && item.run_id === focusTask.id)
    : [];
  const progress = metrics.data?.progress;
  const refreshing = [status, datasets, artifacts, runs, history, config].some((query) => query.isFetching);
  const quickSteps = quickStart.data?.steps || [];
  const smokeDone = quickSteps.some((step) => step.id === "smoke" && step.status === "done");
  const quickStepMap = new Map(quickSteps.map((step) => [step.id, step]));
  const quickStripSteps = ["environment", "model", "dataset", "smoke"].map((id) => quickStepMap.get(id)).filter(Boolean);

  return (
    <div className="overview-page stack">
      <PageToolbar
        icon={Activity}
        title={t("overview")}
        subtitle={t("overviewHint")}
        stats={[
          { icon: Clock3, label: t("status"), value: liveJob ? statusLabel(focusTask, t) : t("status_idle") },
          { icon: Database, label: t("datasetProfiles"), value: `${readyProfiles}/${profiles.length}` },
          { icon: Cpu, label: t("registeredModels"), value: `${usableModels}/${models.length}` },
          { icon: Box, label: t("allArtifacts"), value: artifactItems.filter((item) => !item.archived).length },
        ]}
        onRefresh={() => queryClient.invalidateQueries()}
        refreshing={refreshing}
        refreshLabel={t("refresh")}
      />
      {quickStart.data && !smokeDone && (
        <section className="overview-quickstart-strip">
          <div className="overview-quickstart-copy">
            <span><MapIcon size={17} /> {t("continueQuickStart")}</span>
            <strong>{t("quickStartOverviewTitle")}</strong>
            <small>{t("quickStartOverviewHint")}</small>
          </div>
          <div className="overview-quickstart-steps">
            {quickStripSteps.map((step) => <QuickStartStep step={step} t={t} key={step.id} />)}
          </div>
          <div className="overview-quickstart-actions">
            <button className="golden-primary" onClick={() => goTo(quickStart.data?.next_step?.route || "golden")}>
              {t("goldenFixNextStep")} <ArrowRight size={15} />
            </button>
            <button className="golden-secondary" onClick={() => goTo("golden")}>
              {t("viewQuickStart")}
            </button>
          </div>
        </section>
      )}
      <div className="overview-workspace">
      <section className="overview-focus">
        <div className="overview-focus-main">
          <div className="overview-section-heading">
            <div>
              <h2>{t("currentTraining")}</h2>
              <p>{focusTask ? focusTask.id : t("noActiveTrainingTask")}</p>
            </div>
            <span className={`status-pill ${focusTask?.status || "idle"}`}>{statusLabel(focusTask, t)}</span>
          </div>

          {focusTask ? (
            <>
              <div className="overview-task-summary">
                <div><span>{t("baseModel")}</span><strong>{focusTask.raw?.model_id || "-"}</strong></div>
                <div><span>{t("datasetProfile")}</span><strong>{focusTask.datasetProfile || "-"}</strong></div>
                <div><span>{t("mode")}</span><strong>{focusTask.mode ? t(focusTask.mode) : "-"}</strong></div>
                <div><span>{t("elapsed")}</span><strong>{durationText(focusTask.started, focusTask.finished, focusTask.elapsedSeconds)}</strong></div>
                <div><span>{t("artifactCount")}</span><strong>{taskArtifacts.length}</strong></div>
              </div>
              {progress?.percent != null && (
                <div className="overview-progress">
                  <div><span>{t("trainingStep")}</span><strong>{progress.current_step || 0} / {progress.total_steps || "-"}</strong></div>
                  <i><b style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} /></i>
                </div>
              )}
            </>
          ) : (
            <EmptyState text={t("overviewNoActiveTraining")} />
          )}

          <div className="overview-focus-actions">
            <button className={liveJob ? "primary-button" : "secondary-button"} onClick={() => goTo("training")}>
              {liveJob ? <Play size={16} /> : null}
              {liveJob ? t("viewTraining") : t("enterTrainingPage")}
              {!liveJob ? <ArrowRight size={15} /> : null}
            </button>
            {focusTask && (
              <button className="secondary-button" onClick={() => goTo("history")}>
                {t("viewTaskDetails")} <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>

        <div className="overview-loss-summary">
          <div>
            <span>{t("taskLoss")}</span>
            <small>{lossPoints.length ? `${lossPoints.length} ${t("metricPoints")}` : t("noLossData")}</small>
          </div>
          <strong>{formatNumber(latestLoss, 4)}</strong>
          {lossChange != null && lossPoints.length > 1 ? (
            <span className={lossChange <= 0 ? "loss-good" : "loss-bad"}>
              {lossChange <= 0 ? "↓" : "↑"} {formatNumber(Math.abs(lossChange), 2)}%
              <small>{t("fromFirstStep")}</small>
            </span>
          ) : (
            <span className="loss-neutral">
              {lossPoints.length === 1 ? `step ${lossPoints[0].step}` : "-"}
              <small>{t("lossBelongsToTask")}</small>
            </span>
          )}
          <div className="overview-loss-range">
            <span>{t("firstValue")}<strong>{formatNumber(firstLoss, 4)}</strong></span>
            <span>{t("latestValue")}<strong>{formatNumber(latestLoss, 4)}</strong></span>
          </div>
        </div>
      </section>

      <section className="overview-metrics">
        <OverviewMetric
          icon={Database}
          label={t("datasetProfiles")}
          value={`${readyProfiles} / ${profiles.length}`}
          detail={!hasProfiles ? t("noDatasetProfiles") : readyProfiles === profiles.length ? t("allProfilesReady") : t("profilesNeedAttention")}
          onClick={() => goTo("corpus")}
        />
        <OverviewMetric
          icon={Gauge}
          label={t("usableModels")}
          value={usableModels}
          detail={models.length ? `${models.length} ${t("modelsUnit")}` : t("noData")}
          onClick={() => goTo("models")}
        />
        <OverviewMetric
          icon={Box}
          label={t("deployableAdapters")}
          value={adapters.length}
          detail={adapters.length ? t("readyForInference") : t("noDeployableAdapters")}
          onClick={() => goTo("artifacts")}
        />
      </section>

      {issues.length > 0 && (
        <section className="overview-attention">
          <header>
            <AlertTriangle size={17} />
            <strong>{t("attentionItems")}</strong>
            <span>{issues.length}</span>
          </header>
          <div>
            {issues.map((issue) => (
              <button key={issue.text} onClick={() => goTo(issue.route)}>
                <span>{issue.text}</span><ArrowRight size={15} />
              </button>
            ))}
          </div>
        </section>
      )}

      <div className="overview-activity-grid">
        <Panel
          title={t("recentTrainingTasks")}
          subtitle={t("recentTrainingTasksHint")}
          actions={<button className="secondary-button" onClick={() => goTo("history")}>{t("viewAll")} <ArrowRight size={15} /></button>}
        >
          <div className="overview-task-list">
            {recentTasks.map((task) => <TaskRow task={task} artifacts={artifactItems} t={t} key={task.id} />)}
            {!recentTasks.length && <EmptyState text={t("noTrainingHistory")} />}
          </div>
        </Panel>

        <Panel
          title={t("recentArtifacts")}
          subtitle={t("recentArtifactsHint")}
          actions={<button className="secondary-button" onClick={() => goTo("artifacts")}>{t("viewAll")} <ArrowRight size={15} /></button>}
        >
          <div className="overview-artifact-list">
            {recentArtifacts.map((item) => <ArtifactRow item={item} t={t} key={item.path} />)}
            {!recentArtifacts.length && <EmptyState text={t("noData")} />}
          </div>
        </Panel>
      </div>

      </div>
    </div>
  );
}
