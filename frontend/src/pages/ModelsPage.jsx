import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Boxes, CheckCircle2, Cpu, FolderPlus, FolderSearch, Loader2, Plus, Search, Trash2 } from "lucide-react";
import { apiDelete, apiErrorMessage, apiGet, apiPost } from "../api.js";
import { ConfirmDialog, PageToolbar, Panel } from "../components/ui.jsx";
import { ModelRecommendations } from "../components/ModelRecommendations.jsx";

function normalizeLocalPath(value) {
  return String(value || "").replaceAll("\\", "/").replace(/^\.\//, "").toLowerCase();
}

function hasRegisteredPath(candidate, models) {
  const candidatePath = normalizeLocalPath(candidate.path);
  return (models || []).some((model) => (
    (model.branches || []).some((branch) => normalizeLocalPath(branch.path) === candidatePath)
  ));
}

function formatModelNumber(value, suffix = "B") {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "-";
  return `${number.toLocaleString(undefined, { maximumFractionDigits: 1 })}${suffix}`;
}

function modelFitLabel(suitability, t) {
  return t(`modelFit_${suitability?.status || "unknown"}`);
}

function modelFitDetail(suitability, t) {
  const status = suitability?.status || "unknown";
  return t(`modelFitDetail_${status}`, {
    params: formatModelNumber(suitability?.params_b),
    estimated: formatModelNumber(suitability?.estimated_vram_gb, " GB"),
    available: formatModelNumber(suitability?.available_vram_gb, " GB"),
  });
}

function ModelFitNotice({ suitability, t }) {
  const status = suitability?.status || "unknown";
  return (
    <div className={`model-fit ${status}`}>
      <strong>{modelFitLabel(suitability, t)}</strong>
      <span>{modelFitDetail(suitability, t)}</span>
    </div>
  );
}

function ggufFileCount(notices) {
  return (notices || [])
    .filter((item) => item.format === "gguf")
    .reduce((total, item) => total + Number(item.file_count || 0), 0);
}

function GgufNoticeList({ notices, t }) {
  const ggufNotices = (notices || []).filter((item) => item.format === "gguf");
  if (!ggufNotices.length) return null;
  return (
    <div className="gguf-notice-list">
      <header>
        <strong>{t("ggufDetectedTitle")}</strong>
        <span>{t("ggufDetectedHint", { count: ggufFileCount(ggufNotices) })}</span>
      </header>
      <div className="model-usable-guide">
        <div>
          <strong>{t("usableModelGuideTitle")}</strong>
          <p>{t("usableModelGuideHint")}</p>
        </div>
        <ul>
          <li>{t("usableModelGuideConfig")}</li>
          <li>{t("usableModelGuideTokenizer")}</li>
          <li>{t("usableModelGuideWeights")}</li>
        </ul>
        <p>{t("unsupportedModelGuideHint")}</p>
      </div>
      {ggufNotices.map((item) => (
        <article className="gguf-notice-row" key={item.path}>
          <div>
            <strong>{item.path}</strong>
            <small>{(item.files || []).join(", ") || "-"}</small>
          </div>
          <span className="status-pill repair-manual">{item.file_count} {t("filesUnit")}</span>
        </article>
      ))}
    </div>
  );
}

export function ModelsPage({ t, lang }) {
  const queryClient = useQueryClient();
  const config = useQuery({ queryKey: ["config"], queryFn: () => apiGet("/api/config") });
  const [scanLabel, setScanLabel] = useState("");
  const [removeCandidate, setRemoveCandidate] = useState(null);
  const scan = useMutation({
    mutationFn: (payload) => apiPost("/api/models/scan", payload),
    onSuccess: (_data, payload) => setScanLabel(payload?.all ? t("allModelDirs") : payload?.root || ""),
  });
  const directoryMutation = useMutation({
    mutationFn: (payload) => apiPost("/api/models/directories", payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["config"] }),
  });
  const directoryPicker = useMutation({
    mutationFn: async () => {
      const selected = await apiPost("/api/models/select-directory", {});
      if (!selected.path) return selected;
      await apiPost("/api/models/directories", { action: "add", path: selected.path });
      const scanResult = await apiPost("/api/models/scan", { root: selected.path });
      return { ...selected, scan: scanResult };
    },
    onSuccess: (data) => {
      if (!data?.cancelled) {
        setScanLabel(data.path || "");
        queryClient.invalidateQueries({ queryKey: ["config"] });
      }
    },
  });
  const register = useMutation({
    mutationFn: (candidate) => apiPost("/api/models/register", { candidate, make_active: true }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["config"] }),
  });
  const removeModel = useMutation({
    mutationFn: (modelId) => apiDelete(`/api/models/${encodeURIComponent(modelId)}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["config"] }),
  });

  const models = config.data?.models || [];
  const modelDirs = config.data?.model_scan_dirs || [];
  const scanResult = scan.data || directoryPicker.data?.scan;
  const scanCandidates = scanResult?.candidates || [];
  const scanNotices = scanResult?.notices || [];
  const scanCount = scanResult?.scans?.length || 0;
  const visitedDirs = scanResult?.visited_dirs ?? (scanResult?.scans || []).reduce((total, item) => total + Number(item.visited_dirs || 0), 0);
  const scanErrors = (scanResult?.scans || []).filter((item) => item.ok === false);
  const usableModels = models.filter((model) => (model.branches || []).some((branch) => branch.path_exists)).length;

  return (
    <div className="models-page stack">
      <PageToolbar
        icon={Cpu}
        title={t("models")}
        subtitle={t("modelsHint")}
        stats={[
          { icon: FolderSearch, label: t("modelDirs"), value: modelDirs.length },
          { icon: Boxes, label: t("registeredModels"), value: `${usableModels}/${models.length}` },
        ]}
        onRefresh={() => queryClient.invalidateQueries({ queryKey: ["config"] })}
        refreshing={config.isFetching}
        refreshLabel={t("refresh")}
      />

      <div className="models-workspace">
        <Panel
          title={t("modelDirectories")}
          subtitle={t("modelDirectoriesHint")}
          actions={(
            <button className="secondary-button" onClick={() => scan.mutate({ all: true })} disabled={scan.isPending || !modelDirs.length}>
              {scan.isPending ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
              {t("scanAllModelDirs")}
            </button>
          )}
        >
          <div className="model-scan">
            <div className="model-dir-picker">
              <button className="secondary-button" disabled={directoryPicker.isPending} onClick={() => directoryPicker.mutate()}>
                {directoryPicker.isPending ? <Loader2 className="spin" size={16} /> : <FolderPlus size={16} />}
                {t("chooseAndAddModelDir")}
              </button>
              <span>{t("chooseModelDirHint")}</span>
              <div className="model-dir-feedback" role="status">
                {directoryPicker.error && <span className="error-text">{apiErrorMessage(directoryPicker.error, t)}</span>}
                {directoryPicker.data?.scan && !directoryPicker.error && (
                  <span className="success-text">{t("modelDirAddedAndScanned", { count: directoryPicker.data.scan.candidates?.length || 0 })}</span>
                )}
              </div>
            </div>
            <div className="model-dir-list">
              <header>
                <h3>{t("modelDirs")}</h3>
                <span>{modelDirs.length} {t("foldersUnit")}</span>
              </header>
              {modelDirs.map((item) => (
                <article className={`model-dir-row ${item.exists ? "" : "invalid"}`} key={item.path_resolved || item.path}>
                  <div>
                    <strong>{item.path}</strong>
                    <small>{item.path_resolved}</small>
                  </div>
                  {!item.exists && <span className="status-pill failed">{t("missing")}</span>}
                  <button className="secondary-button" disabled={!item.exists || scan.isPending} onClick={() => scan.mutate({ root: item.path })}>
                    <Search size={16} /> {t("scanModels")}
                  </button>
                  <button className="secondary-button danger-lite" disabled={directoryMutation.isPending} onClick={() => directoryMutation.mutate({ action: "remove", path: item.path })}>
                    <Trash2 size={16} /> {t("remove")}
                  </button>
                </article>
              ))}
              {!modelDirs.length && <div className="empty-state">{t("noModelDirs")}</div>}
              {directoryMutation.error && <div className="error-text">{apiErrorMessage(directoryMutation.error, t)}</div>}
            </div>
          </div>
        </Panel>

        <Panel title={t("registeredModels")} subtitle={t("registeredModelsHint")}>
          <div className="model-catalog">
            {models.map((model) => (
              <article className="model-card" key={model.id}>
                <header>
                  <strong>{model.name || model.id}</strong>
                  <div className="model-card-actions">
                    {model.active && <span className="status-pill completed">{t("active")}</span>}
                    <button
                      className="secondary-button danger-lite compact-button"
                      disabled={removeModel.isPending}
                      onClick={() => setRemoveCandidate(model)}
                    >
                      <Trash2 size={14} /> {t("remove")}
                    </button>
                  </div>
                </header>
                <small>{model.description || model.id}</small>
                <div className="model-card-context">{t("modelLoadMethodsHint")}</div>
                <ModelFitNotice suitability={model.suitability} t={t} />
                <div className="model-branch-list" aria-label={t("loadMethods")}>
                  {(model.branches || []).map((branch) => (
                    <div className={`model-branch-row ${branch.path_exists ? "available" : "missing"}`} key={branch.id}>
                      <div className="model-branch-main">
                        <strong>{branch.id}</strong>
                        <small title={branch.path || ""}>{branch.path || "-"}</small>
                      </div>
                      <span className={`status-pill ${branch.path_exists ? "completed" : "failed"}`}>
                        {branch.path_exists ? t("loadMethodAvailable") : t("loadMethodMissing")}
                      </span>
                    </div>
                  ))}
                  {!model.branches?.length && <div className="empty-state compact">{t("modelNoUsableBranch")}</div>}
                </div>
              </article>
            ))}
            {!models.length && <div className="empty-state">{t("noRegisteredModels")}</div>}
            {removeModel.error && <div className="error-text">{apiErrorMessage(removeModel.error, t)}</div>}
          </div>
        </Panel>
      </div>

      {scanResult && (
        <Panel title={t("scanResults")} subtitle={t("scanResultsHint")}>
          <div className="model-candidates">
            <header>
              <h3>{t("trainableModelDirs")}</h3>
              <span>{scanLabel} · {scanCandidates.length} {t("modelsUnit")} · {visitedDirs} {t("foldersUnit")}{scanCount ? ` · ${scanCount} ${t("scanDirsUnit")}` : ""}</span>
            </header>
            {scanErrors.length > 0 && (
              <div className="scan-error-list">
                {scanErrors.map((item) => <span key={item.root}>{item.root}: {item.error}</span>)}
              </div>
            )}
            <div className="candidate-list">
              {scanCandidates.map((candidate) => {
                const registered = hasRegisteredPath(candidate, models);
                return (
                  <article className="candidate-card" key={candidate.path}>
                    <div>
                      <strong>{candidate.name}</strong>
                      <small>{candidate.path}</small>
                      <span>{candidate.model_type || "-"} · {candidate.quant_format || "-"} · {candidate.weight_file_count} {t("weightFiles")}</span>
                      <ModelFitNotice suitability={candidate.suitability} t={t} />
                    </div>
                    <button className="secondary-button" disabled={registered || register.isPending} onClick={() => register.mutate(candidate)}>
                      {registered ? <CheckCircle2 size={16} /> : <Plus size={16} />}
                      {registered ? t("registered") : t("registerModel")}
                    </button>
                  </article>
                );
              })}
              {!scanCandidates.length && <div className="empty-state">{t("noModelsFound")}</div>}
            </div>
            <GgufNoticeList notices={scanNotices} t={t} />
            {register.error && <div className="error-text">{apiErrorMessage(register.error, t)}</div>}
          </div>
        </Panel>
      )}

      {!usableModels && <ModelRecommendations t={t} lang={lang} />}
      <ConfirmDialog
        open={Boolean(removeCandidate)}
        title={t("removeModelTitle")}
        message={removeCandidate ? t("removeModelConfirm", { name: removeCandidate.name || removeCandidate.id }) : ""}
        confirmLabel={t("confirmRemove")}
        cancelLabel={t("cancel")}
        pending={removeModel.isPending}
        onCancel={() => setRemoveCandidate(null)}
        onConfirm={() => {
          if (!removeCandidate) return;
          removeModel.mutate(removeCandidate.id);
          setRemoveCandidate(null);
        }}
      />
    </div>
  );
}
