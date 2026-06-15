import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Loader2, PackageCheck, RefreshCw, Settings, ShieldCheck, Wrench, X } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost } from "../api.js";
import { PageToolbar, Panel, PropertyList } from "../components/ui.jsx";

function dependencyStatusLabel(status, t) {
  return {
    ready: t("dependencyReady"),
    missing: t("dependencyMissing"),
    incompatible: t("dependencyIncompatible"),
    optional: t("dependencyOptional"),
  }[status] || status;
}

function dependencyRepairModeLabel(mode, t) {
  return {
    auto: t("dependencyRepairAuto"),
    launcher: t("dependencyRepairLauncher"),
    manual: t("dependencyRepairManual"),
  }[mode] || t("dependencyRepairUnknown");
}

function dependencyRepairClass(mode) {
  return `repair-${mode || "unknown"}`;
}

const DEPENDENCY_GROUPS = [
  { id: "runtime", title: "dependencyGroupRuntime", ids: ["python"] },
  { id: "training", title: "dependencyGroupTraining", ids: ["torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes"] },
  { id: "compute", title: "dependencyGroupCompute", ids: ["compute_backend", "cuda", "nvidia_driver"] },
  { id: "frontend", title: "dependencyGroupFrontend", ids: ["node", "npm"] },
  { id: "optional", title: "dependencyGroupOptional", ids: ["unsloth"] },
];

function groupedDependencies(items = []) {
  const byId = new Map(items.map((item) => [item.id, item]));
  const used = new Set();
  const groups = DEPENDENCY_GROUPS.map((group) => {
    const groupItems = group.ids.map((id) => byId.get(id)).filter(Boolean);
    groupItems.forEach((item) => used.add(item.id));
    return { ...group, items: groupItems };
  }).filter((group) => group.items.length);
  const otherItems = items.filter((item) => !used.has(item.id));
  if (otherItems.length) {
    groups.push({ id: "other", title: "dependencyGroupOther", items: otherItems });
  }
  return groups;
}

function dependencyGroupCount(items) {
  const required = items.filter((item) => item.required);
  const total = required.length || items.length;
  const ready = (required.length ? required : items).filter((item) => ["ready", "optional"].includes(item.status)).length;
  return `${ready}/${total}`;
}

function trainingDependencyCount(dependencies) {
  const items = dependencies?.items || [];
  const frontendIds = new Set(["node", "npm"]);
  const required = items.filter((item) => item.required && !frontendIds.has(item.id));
  return {
    ready: required.filter((item) => item.status === "ready").length,
    total: required.length,
  };
}

function currentEnvironmentSummary(t, dependencies) {
  if (!dependencies) {
    return { status: "checking", label: t("currentEnvironmentChecking"), detail: t("currentEnvironmentCheckingDetail") };
  }
  const accelerator = dependencies?.accelerator || {};
  const platform = dependencies?.platform || {};
  const backend = accelerator.backend || "unknown";
  const system = String(platform.system || "").toLowerCase();
  const counts = trainingDependencyCount(dependencies);
  const requiredReady = counts.ready;
  const requiredTotal = counts.total;
  const dependencyReady = requiredTotal > 0 && requiredReady === requiredTotal;
  const isWindows = system.includes("windows");
  const isVerified = backend === "cuda" && isWindows && dependencyReady;
  const isCuda = backend === "cuda";

  if (isVerified) {
    return { status: "supported", label: t("currentEnvironmentSupported"), detail: t("currentEnvironmentVerifiedDetail") };
  }
  if (isCuda && !dependencyReady) {
    return { status: "pending", label: t("currentEnvironmentNeedsCheck"), detail: t("currentEnvironmentMissingDetail") };
  }
  if (isCuda) {
    return { status: "pending", label: t("currentEnvironmentNeedsCheck"), detail: t("currentEnvironmentCudaUnverifiedDetail") };
  }
  return { status: "unsupported", label: t("currentEnvironmentNotSupported"), detail: t("currentEnvironmentUnsupportedDetail") };
}

function DoctorDialog({ open, doctor, onClose, t }) {
  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="command-dialog-backdrop" onMouseDown={onClose}>
      <section className="command-dialog" role="dialog" aria-modal="true" aria-label={t("environmentCheck")} onMouseDown={(event) => event.stopPropagation()}>
        <header className="command-dialog-head">
          <div>
            <h2>{t("environmentCheck")}</h2>
            <p>{t("doctorHint")}</p>
          </div>
          <button className="secondary-button icon-only" title={t("close")} onClick={onClose}>
            <X size={18} />
          </button>
        </header>
        {doctor.error && <div className="error-text">{apiErrorMessage(doctor.error, t)}</div>}
        <pre className="doctor-output">
          {doctor.isPending ? t("checkingDependencies") : doctor.data ? `${doctor.data.stdout || ""}\n${doctor.data.stderr || ""}`.trim() : t("doctorHint")}
        </pre>
      </section>
    </div>
  );
}

function DependencyPanel({ dependencies, doctor, repair, t }) {
  const [doctorOpen, setDoctorOpen] = useState(false);

  const openDoctor = () => {
    setDoctorOpen(true);
    doctor.mutate();
  };

  return (
    <>
      <Panel
        title={t("systemDependencies")}
        subtitle={t("systemDependenciesHint")}
        actions={(
          <>
            <button className="secondary-button" onClick={openDoctor} disabled={doctor.isPending}>
              {doctor.isPending ? <Loader2 className="spin" size={16} /> : <CheckCircle2 size={16} />}
              {t("runDoctor")}
            </button>
            <button className="secondary-button" onClick={() => repair.mutate()} disabled={repair.isPending}>
              {repair.isPending ? <Loader2 className="spin" size={16} /> : <Wrench size={16} />}
              {t("repairAutoDependencies")}
            </button>
            <button className="secondary-button" onClick={() => dependencies.refetch()} disabled={dependencies.isFetching}>
              {dependencies.isFetching ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
              {t("checkAgain")}
            </button>
          </>
        )}
      >
        {dependencies.isLoading && <div className="dependency-loading"><Loader2 className="spin" size={18} /> {t("checkingDependencies")}</div>}
        {dependencies.error && <div className="error-text">{apiErrorMessage(dependencies.error, t)}</div>}
        {dependencies.data && (
          <div className="dependency-groups">
            <div className="dependency-repair-guide">
              <strong>{t("dependencyRepairScope")}</strong>
              <span>{t("dependencyRepairScopeHint")}</span>
            </div>
            {repair.error && <div className="error-text">{apiErrorMessage(repair.error, t)}</div>}
            {repair.data && (
              <div className="dependency-repair-result">
                <strong>{t("dependencyRepairResult")}</strong>
                <pre>{`${repair.data.stdout || ""}\n${repair.data.stderr || ""}`.trim() || t("dependencyRepairNoOutput")}</pre>
              </div>
            )}
            {groupedDependencies(dependencies.data.items || []).map((group) => (
              <div className="dependency-group" key={group.id}>
                <header className="dependency-group-head">
                  <strong>{t(group.title)}</strong>
                  <span>{dependencyGroupCount(group.items)}</span>
                </header>
                <div className="dependency-list">
                  <div className="dependency-table-head" aria-hidden="true">
                    <span>{t("dependencyColumnName")}</span>
                    <span>{t("dependencyColumnVersion")}</span>
                    <span>{t("dependencyColumnRequirement")}</span>
                    <span>{t("dependencyColumnStatus")}</span>
                    <span>{t("dependencyColumnRepair")}</span>
                  </div>
                  {group.items.map((item) => (
                    <div className="dependency-row" key={item.id}>
                      <div>
                        <strong>{item.name}</strong>
                        <small title={item.detail || ""}>{item.detail || (item.required ? t("requiredDependency") : t("optionalDependency"))}</small>
                        <small className="dependency-repair-hint" title={t(item.repair?.hint || "repairUnknownHint")}>
                          {t(item.repair?.hint || "repairUnknownHint")}
                        </small>
                      </div>
                      <span>{item.version || "-"}</span>
                      <small>{item.requirement || "-"}</small>
                      <span className={`status-pill ${item.status === "ready" ? "completed" : item.status === "optional" ? "" : "failed"}`}>
                        {dependencyStatusLabel(item.status, t)}
                      </span>
                      <span className={`status-pill repair-mode ${dependencyRepairClass(item.repair?.mode)}`}>
                        {dependencyRepairModeLabel(item.repair?.mode, t)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>
      <DoctorDialog open={doctorOpen} doctor={doctor} onClose={() => setDoctorOpen(false)} t={t} />
    </>
  );
}

export function EnvironmentPage({ t }) {
  const queryClient = useQueryClient();
  const config = useQuery({ queryKey: ["config"], queryFn: () => apiGet("/api/config") });
  const dependencies = useQuery({
    queryKey: ["environment-dependencies"],
    queryFn: () => apiGet("/api/environment/dependencies"),
  });
  const doctor = useMutation({ mutationFn: () => apiGet("/api/doctor") });
  const repair = useMutation({
    mutationFn: () => apiPost("/api/environment/repair", {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["environment-dependencies"] });
      queryClient.invalidateQueries({ queryKey: ["config"] });
    },
  });
  const environmentSummary = currentEnvironmentSummary(t, dependencies.data);
  const trainingDependencies = trainingDependencyCount(dependencies.data);

  return (
    <div className="environment-page stack">
      <PageToolbar
        icon={Settings}
        title={t("environment")}
        subtitle={t("envHint")}
        stats={[
          { icon: PackageCheck, label: t("availableDependencies"), value: dependencies.data ? `${dependencies.data.counts?.required_ready || 0}/${dependencies.data.counts?.required_total || 0}` : "-" },
        ]}
        onRefresh={() => {
          queryClient.invalidateQueries({ queryKey: ["config"] });
          queryClient.invalidateQueries({ queryKey: ["environment-dependencies"] });
        }}
        refreshing={config.isFetching || dependencies.isFetching}
        refreshLabel={t("refresh")}
      />
      <div className="environment-columns">
        <Panel title={t("trainingEnvironmentStatus")} subtitle={t("trainingEnvironmentStatusHint")}>
          <div className="training-support-summary">
            {dependencies.isLoading ? <Loader2 className="spin" size={18} /> : <ShieldCheck size={18} />}
            <div>
              <strong>{environmentSummary.label}</strong>
              <span>{environmentSummary.detail}</span>
            </div>
          </div>
          <div className={`current-environment-card ${environmentSummary.status}`}>
            <PropertyList rows={[
              [t("detectedOs"), dependencies.data?.platform?.system || "-"],
              [t("detectedAccelerator"), dependencies.data?.accelerator?.device_name || t("notDetected")],
              [t("detectedBackend"), dependencies.data?.accelerator?.backend || "-"],
              [t("dependencyReadiness"), dependencies.data ? `${trainingDependencies.ready}/${trainingDependencies.total}` : "-"],
            ]} />
          </div>
        </Panel>
        <Panel title={t("projectAndServices")} subtitle={t("projectAndServicesHint")}>
          <PropertyList rows={[
            [t("projectRoot"), config.data?.project_root],
            [t("configFile"), config.data?.config_file],
            [t("projectVersion"), config.data?.project?.version],
            [t("dashboardDefaultPort"), config.data?.monitoring?.dashboard_port],
          ]} />
        </Panel>
      </div>
      <DependencyPanel dependencies={dependencies} doctor={doctor} repair={repair} t={t} />
    </div>
  );
}
