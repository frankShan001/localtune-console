import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Clipboard, Download, Loader2, Square } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatDate, formatNumber } from "../api.js";
import { OpenFolderButton } from "./ui.jsx";

function modelFitKey(status) {
  return `recommendedModelFit_${status || "unknown"}`;
}

function runningJob(job) {
  return job && ["running", "stopping"].includes(job.status);
}

export function ModelRecommendations({ t, lang = "en", compact = false }) {
  const queryClient = useQueryClient();
  const [copiedId, setCopiedId] = useState("");
  const recommendations = useQuery({
    queryKey: ["model-recommendations", lang],
    queryFn: () => apiGet(`/api/models/recommendations?locale=${encodeURIComponent(lang)}`),
    refetchInterval: 6000,
  });
  const startDownload = useMutation({
    mutationFn: (item) => apiPost("/api/models/downloads/start", {
      id: item.id,
      provider: item.provider,
      provider_model_id: item.provider_model_id,
      target_dir: recommendations.data?.target_dir || "models",
    }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-recommendations", lang] }),
  });
  const cancelDownload = useMutation({
    mutationFn: (jobId) => apiPost(`/api/models/downloads/${encodeURIComponent(jobId)}/cancel`, {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-recommendations", lang] }),
  });

  const copyCommand = async (item) => {
    try {
      await navigator.clipboard?.writeText(item.download_command);
      setCopiedId(item.id);
      window.setTimeout(() => setCopiedId(""), 1800);
    } catch {
      setCopiedId("");
    }
  };

  if (recommendations.isLoading && !recommendations.data) {
    return (
      <section className={`model-recommendations ${compact ? "compact" : ""}`}>
        <div className="model-recommendations-loading"><Loader2 className="spin" size={18} /> {t("loadingModelRecommendations")}</div>
      </section>
    );
  }

  if (recommendations.error) {
    return (
      <section className={`model-recommendations ${compact ? "compact" : ""}`}>
        <div className="error-text">{apiErrorMessage(recommendations.error, t)}</div>
      </section>
    );
  }

  const items = recommendations.data?.recommendations || [];
  const visibleItems = compact ? items.slice(0, 4) : items;
  const accelerator = recommendations.data?.accelerator || {};
  const providerLabel = recommendations.data?.provider === "modelscope" ? "ModelScope" : "Hugging Face";

  return (
    <section className={`model-recommendations ${compact ? "compact" : ""}`}>
      <header className="model-recommendations-head">
        <div>
          <span>{t("recommendedModelsLabel")}</span>
          <h2>{t("recommendedModelsTitle")}</h2>
          <p>{t("recommendedModelsHint", {
            device: accelerator.device_name || t("notDetected"),
            vram: accelerator.max_memory ? `${formatNumber(accelerator.max_memory, 1)} GB` : "-",
            provider: providerLabel,
          })}</p>
        </div>
        <button className="secondary-button" onClick={() => recommendations.refetch()}>
          {recommendations.isFetching ? <Loader2 className="spin" size={16} /> : null}
          {t("refresh")}
        </button>
      </header>

      <div className="recommendation-grid">
        {visibleItems.map((item) => {
          const job = item.download_job;
          const isRunning = runningJob(job);
          const downloadedPath = job?.status === "completed" ? job.model_path : "";
          return (
            <article className={`recommendation-card ${item.fit?.status || "unknown"}`} key={item.id}>
              <div className="recommendation-card-head">
                <div>
                  <strong>{item.name}</strong>
                </div>
                <span className={`recommendation-fit ${item.fit?.status || "unknown"}`}>
                  {t(modelFitKey(item.fit?.status))}
                </span>
              </div>
              <p>{item.summary}</p>
              <div className="recommendation-meta">
                <span>{t("recommendedMinVram")}: {formatNumber(item.min_vram_gb, 1)} GB</span>
                <span>{t("recommendedProvider")}: {providerLabel}</span>
              </div>
              {job && (
                <div className={`download-status ${job.status}`}>
                  <strong>{t(`downloadStatus_${job.status}`)}</strong>
                  <span>{job.started_at ? formatDate(job.started_at) : ""}</span>
                  {downloadedPath ? (
                    <small>{downloadedPath}</small>
                  ) : null}
                </div>
              )}
              {!item.download_available && (
                <div className="download-status warning">
                  <strong>{t("downloadDependencyMissing")}</strong>
                  <span>{t("downloadDependencyMissingHint")}</span>
                </div>
              )}
              <div className="download-actions">
                <a className="secondary-button" href={item.download_url} target="_blank" rel="noreferrer">
                  <ArrowUpRight size={15} /> {t("openDownloadPage")}
                </a>
                <button className="secondary-button" onClick={() => copyCommand(item)}>
                  <Clipboard size={15} /> {copiedId === item.id ? t("copied") : t("copyDownloadCommand")}
                </button>
                {downloadedPath ? (
                  <OpenFolderButton path={downloadedPath} label={t("openFolder")} />
                ) : isRunning ? (
                  <button className="secondary-button danger-lite" disabled={cancelDownload.isPending} onClick={() => cancelDownload.mutate(job.id)}>
                    <Square size={14} /> {t("cancelDownload")}
                  </button>
                ) : (
                  <button className="secondary-button" disabled={startDownload.isPending || !item.download_available} onClick={() => startDownload.mutate(item)}>
                    {startDownload.isPending ? <Loader2 className="spin" size={15} /> : <Download size={15} />}
                    {t("startDownload")}
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>
      {(startDownload.error || cancelDownload.error) && (
        <div className="error-text">{apiErrorMessage(startDownload.error || cancelDownload.error, t)}</div>
      )}
    </section>
  );
}
