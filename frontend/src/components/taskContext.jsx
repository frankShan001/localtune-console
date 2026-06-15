import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Archive, CheckCircle2, Clock3, Copy, XCircle } from "lucide-react";
import { apiGet, formatBytes, formatDate, shortPath } from "../api.js";
import { LogViewer } from "./monitoring.jsx";
import { ContextDrawer, EmptyState, OpenFolderButton, Panel, PropertyList } from "./ui.jsx";
import { artifactHasExactTaskSource, durationText, taskKindLabel, taskTitle } from "../pages/taskSupport.js";

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

function artifactRelation(item, task) {
  return artifactHasExactTaskSource(item, task) ? "exactRelation" : "";
}

function taskRows(task, t) {
  return [
    [t("runKind"), taskKindLabel(task, t)],
    [t("mode"), task?.mode ? t(task.mode) : "-"],
    [t("branch"), task?.branch],
    [t("dataset"), task?.datasetProfile],
    [t("started"), formatDate(task?.started)],
    [t("finished"), formatDate(task?.finished)],
    [t("elapsed"), durationText(task?.started, task?.finished, task?.elapsedSeconds)],
    [t("returnCode"), task?.returncode],
    [t("outputDir"), task?.outputDir],
    [t("logFile"), task?.file],
  ];
}

export function TaskContextDrawer({ open, onClose, task, artifacts = [], t }) {
  const log = useQuery({
    queryKey: ["context-task-log", task?.file],
    queryFn: () => apiGet(`/api/logs?n=900&file=${encodeURIComponent(task.file)}`),
    enabled: Boolean(open && task?.file),
  });

  const StatusIcon = statusIcon(task?.status);
  const relatedArtifacts = artifacts
    .map((item) => ({ item, relation: artifactRelation(item, task) }))
    .filter(({ relation }) => relation)
    .sort((a, b) => String(b.item.updated || "").localeCompare(String(a.item.updated || "")))
    .slice(0, 8);

  return (
    <ContextDrawer
      open={open}
      onClose={onClose}
      t={t}
      title={t("taskContext")}
      subtitle={task ? taskTitle(task, t) : t("noTaskSelected")}
      actions={task && (
        <>
          <button className="secondary-button" onClick={() => navigator.clipboard?.writeText(task.id)}>
            <Copy size={15} /> {t("copyId")}
          </button>
          <OpenFolderButton path={task.outputDir || task.file} label={t("openFolder")} />
        </>
      )}
    >
      {!task && <EmptyState text={t("noTaskSelected")} />}
      {task && (
        <div className="drawer-stack">
          <Panel title={t("taskSummary")} subtitle={task.id}>
            <div className="task-detail-head">
              <div className={`status-pill ${task.status}`}>
                <StatusIcon size={16} /> {t(`status_${task.status}`)}
              </div>
            </div>
            <PropertyList rows={taskRows(task, t)} />
            {task.raw?.diagnostics && (
              <div className="diagnostic-alert">
                <strong>{task.raw.diagnostics.title}</strong>
                <span>{task.raw.diagnostics.summary}</span>
                {(task.raw.diagnostics.suggestions || []).map((suggestion) => <small key={suggestion}>{suggestion}</small>)}
              </div>
            )}
            {task.source === "log" && <p className="muted-note">{t("legacyTaskHint")}</p>}
          </Panel>

          <Panel title={t("taskParams")}>
            {Object.keys(task.params || {}).length ? (
              <pre className="compact-code">{JSON.stringify(task.params, null, 2)}</pre>
            ) : (
              <EmptyState text={t("noData")} />
            )}
          </Panel>

          <Panel title={t("relatedArtifacts")} subtitle={task.branch || ""}>
            <div className="related-artifacts">
              {relatedArtifacts.map(({ item, relation }) => (
                <article className="related-artifact" key={item.path}>
                  <Archive size={16} />
                  <span>
                    <strong>{artifactLabel(item.type, t)}</strong>
                    <small>{t(relation)} · {shortPath(item.path)} · {formatBytes(item.size_bytes)}</small>
                  </span>
                  <OpenFolderButton path={item.path} label={t("openFolder")} />
                </article>
              ))}
              {!relatedArtifacts.length && <EmptyState text={t("noData")} />}
            </div>
          </Panel>

          <Panel title={t("taskLogPreview")} subtitle={log.data?.file || task.file || ""}>
            {task.file ? <LogViewer lines={log.data?.logs || []} t={t} /> : <EmptyState text={t("noData")} />}
          </Panel>
        </div>
      )}
    </ContextDrawer>
  );
}
