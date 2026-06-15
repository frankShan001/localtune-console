import React, { useEffect, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircuitBoard,
  Cpu,
  Database,
  Loader2,
  Microscope,
  Play,
  RefreshCw,
  Rocket,
  TerminalSquare,
} from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatNumber } from "../api.js";

const stepIcons = {
  environment: Cpu,
  model: CircuitBoard,
  dataset: Database,
  smoke: TerminalSquare,
  train: Rocket,
  evaluate: Microscope,
};

const statusTone = {
  done: "done",
  ready: "ready",
  blocked: "blocked",
};

const GOLDEN_STATUS_CACHE_KEY = "localtune.goldenPathStatus";
const quickStartStepIds = ["environment", "model", "dataset", "smoke", "train", "evaluate"];
const stepFallbackRoutes = {
  environment: "environment",
  model: "models",
  dataset: "corpus",
  smoke: "training",
  train: "training",
  evaluate: "inference",
};

function readCachedGoldenStatus() {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = window.sessionStorage.getItem(GOLDEN_STATUS_CACHE_KEY);
    if (!raw) return undefined;
    const parsed = JSON.parse(raw);
    return parsed && parsed.ok ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function writeCachedGoldenStatus(data) {
  if (typeof window === "undefined" || !data?.ok) return;
  try {
    window.sessionStorage.setItem(GOLDEN_STATUS_CACHE_KEY, JSON.stringify(data));
  } catch {
    // Ignore storage failures; the live query still drives the page.
  }
}

function goTo(route) {
  window.location.hash = `#/${route}`;
}

function compactPath(value, t) {
  if (!value) return t("notSelected");
  const text = String(value);
  return text.length > 48 ? `...${text.slice(-45)}` : text;
}

function activeModelPath(model, branchId) {
  const branch = (model?.branches || []).find((item) => item.id === branchId && item.path_exists)
    || (model?.branches || []).find((item) => item.path_exists);
  return branch?.path || "";
}

function goldenStepLabel(step, t) {
  return t(`goldenStep_${step?.id || "golden"}`);
}

function goldenStepStatus(status, t) {
  return t(`goldenStatus_${status || "blocked"}`);
}

function StepNode({ step, active, t }) {
  const Icon = stepIcons[step.id] || CheckCircle2;
  return (
    <div className={`golden-step ${statusTone[step.status] || ""} ${active ? "active" : ""}`}>
      <span className="golden-step-icon">
        <Icon size={18} />
      </span>
      <span className="golden-step-copy">
        <strong>{goldenStepLabel(step, t)}</strong>
        <small>{goldenStepStatus(step.status, t)}</small>
      </span>
    </div>
  );
}

function stepRoute(step) {
  return step?.route || stepFallbackRoutes[step?.id] || "golden";
}

function RunConfigPanel({ selection, payload, t, canStartSmoke, smokePending, onStartSmoke }) {
  const model = selection?.model;
  const profile = selection?.dataset_profile;
  const branch = selection?.branch;
  const rows = [
    [t("baseModel"), model?.name || t("goldenMissingBaseModelShort")],
    [t("modelPath"), compactPath(activeModelPath(model, branch?.id), t)],
    [t("datasetProfile"), profile?.name || t("selectDatasetProfile")],
    [t("rowsUnit"), formatNumber(profile?.validation?.rows ?? profile?.total_rows ?? 0, 0)],
    [t("loadMethod"), branch?.id || payload?.branch || "bnb4"],
    [t("maxSteps"), payload?.max_steps],
    [t("maxSeqLength"), payload?.max_seq_length],
  ];
  return (
    <section className="golden-run-config">
      <header>
        <div>
          <h2>{t("goldenTestRunConfig")}</h2>
          <p>{t("goldenTestRunConfigHint")}</p>
        </div>
        <button className="golden-primary" disabled={!canStartSmoke || smokePending} onClick={onStartSmoke}>
          {smokePending ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
          {t("goldenStartSmoke")}
        </button>
      </header>
      <div className="golden-run-summary">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong title={String(value || "")}>{value == null || value === "" ? "-" : value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function AfterFixPanel({ t }) {
  const rows = [t("goldenAfterFixStep1"), t("goldenAfterFixStep2"), t("goldenAfterFixStep3")];
  return (
    <section className="golden-after-fix">
      <h2>{t("goldenAfterFixTitle")}</h2>
      <ol>
        {rows.map((row) => <li key={row}>{row}</li>)}
      </ol>
    </section>
  );
}

function TrainabilityPanel({ readiness, t, canStartSmoke, smokePending, onStartSmoke }) {
  const code = readiness?.code || "missing_environment";
  const canTrain = Boolean(readiness?.can_train);
  const route = readiness?.route || "environment";
  const action = canTrain ? onStartSmoke : () => goTo(route);
  const disabled = canTrain && (!canStartSmoke || smokePending);
  return (
    <div className={`golden-trainability ${canTrain ? "ready" : code}`}>
      <span>{canTrain ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}</span>
      <div>
        <small>{t("goldenTrainabilityLabel")}</small>
        <strong>{t(`goldenReadiness_${code}_title`)}</strong>
        <p>{t(`goldenReadiness_${code}_detail`)}</p>
      </div>
      <button className={canTrain ? "golden-primary" : "golden-secondary"} disabled={disabled} onClick={action}>
        {smokePending && canTrain ? <Loader2 className="spin" size={15} /> : null}
        {t(`goldenReadiness_${code}_action`)}
        {!canTrain ? <ArrowRight size={15} /> : null}
      </button>
    </div>
  );
}

export function GoldenRunPage({ t }) {
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: ["golden-path-status"],
    queryFn: () => apiGet("/api/golden-path/status"),
    initialData: readCachedGoldenStatus,
    refetchInterval: 6000,
  });
  useEffect(() => {
    writeCachedGoldenStatus(status.data);
  }, [status.data]);
  const steps = status.data?.steps || [];
  const orderedSteps = useMemo(
    () => quickStartStepIds.map((id) => steps.find((step) => step.id === id)).filter(Boolean),
    [steps],
  );
  const smokePayload = status.data?.payloads?.smoke || {};
  const trainingBusy = ["running", "stopping"].includes(status.data?.training_status?.status);
  const canStartSmoke = Boolean(status.data?.can_start_smoke && !trainingBusy);
  const smoke = useMutation({
    mutationFn: () => apiPost("/api/training/start", smokePayload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["golden-path-status"] });
      queryClient.invalidateQueries({ queryKey: ["training-status"] });
      goTo("training");
    },
  });
  if (status.isLoading && !status.data) {
    return (
      <div className="golden-page">
        <div className="golden-loading"><Loader2 className="spin" size={24} /> {t("goldenLoading")}</div>
      </div>
    );
  }

  if (status.error) {
    return (
      <div className="golden-page">
        <div className="golden-error">
          <AlertTriangle size={22} />
          {apiErrorMessage(status.error, t)}
        </div>
      </div>
    );
  }

  const nextStep = status.data?.next_step || orderedSteps.find((step) => step.status !== "done") || orderedSteps[0];
  return (
    <div className="golden-page">
      <section className="golden-header">
        <div>
          <h1>{t("goldenRun")}</h1>
          <p>{t("goldenRunHint")}</p>
        </div>
        <div className="golden-header-actions">
          <button className="golden-secondary" onClick={() => goTo(stepRoute(nextStep))}>
            {t("goldenFixNextStep")} <ArrowRight size={16} />
          </button>
          <button className="golden-icon-button" title={t("refresh")} onClick={() => status.refetch()}>
            <RefreshCw className={status.isFetching ? "spin" : ""} size={16} />
          </button>
        </div>
      </section>

      <section className="golden-map-section">
        <header>
          <div>
            <h2>{t("goldenPathTitle")}</h2>
            <p>{t("goldenPathHint")}</p>
          </div>
          <strong>{t("goldenNext")}: {goldenStepLabel(nextStep, t)}</strong>
        </header>
        <div className="golden-map">
          {orderedSteps.map((step) => (
            <StepNode key={step.id} step={step} active={nextStep?.id === step.id} t={t} />
          ))}
        </div>
      </section>

      <TrainabilityPanel
        readiness={status.data?.training_readiness}
        t={t}
        canStartSmoke={canStartSmoke}
        smokePending={smoke.isPending}
        onStartSmoke={() => smoke.mutate()}
      />

      {(smoke.error) && (
        <div className="golden-action-error">
          <AlertTriangle size={15} />
          {apiErrorMessage(smoke.error, t)}
        </div>
      )}

      <section className="golden-run-grid">
        <RunConfigPanel
          selection={status.data?.selection}
          payload={smokePayload}
          t={t}
          canStartSmoke={canStartSmoke}
          smokePending={smoke.isPending}
          onStartSmoke={() => smoke.mutate()}
        />
        <AfterFixPanel t={t} />
      </section>
    </div>
  );
}
