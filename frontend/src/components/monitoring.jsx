import React, { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatDate, formatNumber } from "../api.js";
import { EmptyState, SelectControl } from "./ui.jsx";

export function LossChart({ data, t = (key) => key }) {
  const chartData = (data || []).map((item) => ({ step: item.step, loss: Number(item.value) }));
  if (!chartData.length) return <EmptyState text={t("noLossData")} />;
  if (chartData.length === 1) {
    return <div className="single-loss">step {chartData[0].step}<strong>{formatNumber(chartData[0].loss, 4)}</strong><span>{t("singleLossHint")}</span></div>;
  }
  const losses = chartData.map((item) => item.loss).filter(Number.isFinite);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const padding = Math.max((maxLoss - minLoss) * 0.15, 0.01);
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={chartData} margin={{ top: 14, right: 20, left: 22, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis domain={[minLoss - padding, maxLoss + padding]} width={78} tickFormatter={(value) => formatNumber(value, 4)} />
          <Tooltip />
          <Line type="monotone" dataKey="loss" stroke="#2563eb" strokeWidth={2.4} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ResourceBars({ status, t = (key) => key }) {
  const gpu = status?.gpu;
  const system = status?.system;
  return (
    <div className="resource-grid">
      <Meter label={t("cpu")} value={system?.cpu_percent} suffix="%" />
      <Meter label={t("memory")} value={system?.memory_percent} suffix="%" detail={`${formatNumber(system?.memory_used_gb, 1)} / ${formatNumber(system?.memory_total_gb, 1)} GB`} />
      <Meter label={t("gpuMemory")} value={gpu?.max_memory ? (gpu.memory_used / gpu.max_memory) * 100 : null} suffix="%" detail={gpu?.available ? `${formatNumber(gpu.memory_used, 1)} / ${formatNumber(gpu.max_memory, 1)} GB` : gpu?.message || "-"} />
      <Meter label={t("gpuUtil")} value={gpu?.gpu_util} suffix="%" detail={gpu?.device_name} />
      <Meter label={t("gpuTemperature")} value={gpu?.temperature} suffix="°C" detail={gpu?.temperature != null ? t("gpuTemperatureHint") : t("notAvailable")} />
      <Meter
        label={t("gpuPower")}
        value={gpu?.power_limit_w ? (gpu.power_draw_w / gpu.power_limit_w) * 100 : null}
        suffix="%"
        detail={gpu?.power_draw_w != null
          ? gpu?.power_limit_w
            ? `${formatNumber(gpu.power_draw_w, 1)} / ${formatNumber(gpu.power_limit_w, 1)} W`
            : `${formatNumber(gpu.power_draw_w, 1)} W`
          : t("notAvailable")}
      />
    </div>
  );
}

export function Meter({ label, value, suffix, detail }) {
  const numeric = Number(value);
  const valid = Number.isFinite(numeric);
  const clamped = valid ? Math.max(0, Math.min(100, numeric)) : 0;
  return (
    <div className="meter">
      <div>
        <span>{label}</span>
        <strong>{valid ? `${formatNumber(numeric, 1)}${suffix}` : "-"}</strong>
      </div>
      <div className="meter-track"><i style={{ width: `${clamped}%` }} /></div>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function compactDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "-";
  if (value < 60) return `${Math.round(value)} s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  if (minutes < 60) return `${minutes} min ${rest} s`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

export function TaskStatus({ data, progress, t = (key) => key }) {
  const job = data?.job;
  const status = data?.status || "idle";
  const stepText = progress?.total_steps
    ? `${progress.current_step || 0} / ${progress.total_steps}`
    : progress?.current_step || "-";
  const rows = [
    [t("jobId"), job?.id],
    [t("runStatus"), status ? t(`status_${status}`) : "-"],
    [t("mode"), job?.mode],
    [t("trainingStep"), stepText],
    [t("epoch"), progress?.epoch != null ? formatNumber(progress.epoch, 2) : "-"],
    [t("trainingSpeed"), progress?.steps_per_second ? `${formatNumber(progress.steps_per_second, 3)} step/s` : "-"],
    [t("eta"), compactDuration(progress?.eta_seconds)],
    ["PID", job?.pid],
    [t("started"), formatDate(job?.started_at)],
    [t("finished"), formatDate(job?.finished_at)],
  ];
  return (
    <div className="task-status">
      <div className="task-status-head">
        <div className={`status-pill ${status}`}>{t(`status_${status}`)}</div>
        <span>{job?.id || t("noActiveTrainingTask")}</span>
      </div>
      {job ? (
        <>
          {progress?.percent != null && (
            <div className="task-progress">
              <i style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} />
            </div>
          )}
          <div className="task-status-grid">
            {rows.map(([key, value]) => (
              <div className="task-status-item" key={key}>
                <span>{key}</span>
                <strong title={value == null ? "" : String(value)}>{value == null || value === "" ? "-" : String(value)}</strong>
              </div>
            ))}
          </div>
          {job.diagnostics && (
            <div className="diagnostic-alert">
              <strong>{job.diagnostics.title}</strong>
              <span>{job.diagnostics.summary}</span>
              {(job.diagnostics.suggestions || []).map((suggestion) => <small key={suggestion}>{suggestion}</small>)}
            </div>
          )}
        </>
      ) : (
        <div className="task-status-empty">{t("startTrainingToMonitor")}</div>
      )}
    </div>
  );
}

export function LogViewer({ lines, t = (key) => key }) {
  const ref = useRef(null);
  const [follow, setFollow] = useState(true);
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState("all");
  const normalizedQuery = query.trim().toLowerCase();
  const filteredLines = (lines || []).filter((line) => {
    const text = String(line || "");
    const lower = text.toLowerCase();
    if (normalizedQuery && !lower.includes(normalizedQuery)) return false;
    if (level === "error") return /\b(error|failed|exception|traceback|oom)\b/i.test(text);
    if (level === "warning") return /\b(warn|warning)\b/i.test(text);
    if (level === "metrics") return /\[metrics\]|loss|learning_rate|epoch/i.test(text);
    return true;
  });
  useEffect(() => {
    if (follow && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [filteredLines.length, follow]);
  return (
    <div className="log-console">
      <div className="log-toolbar">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("searchLogs")} />
        <SelectControl
          value={level}
          onChange={setLevel}
          ariaLabel={t("allLogs")}
          options={[
            ["all", t("allLogs")],
            ["metrics", t("metricLogs")],
            ["warning", t("warningLogs")],
            ["error", t("errorLogs")],
          ]}
        />
        <label>
          <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />
          {t("followLogs")}
        </label>
        <span>{filteredLines.length} / {(lines || []).length}</span>
      </div>
      <div className="log-viewer" ref={ref} onScroll={() => {
        const el = ref.current;
        if (!el || !follow) return;
        if (el.scrollHeight - el.scrollTop - el.clientHeight >= 32) setFollow(false);
      }}>
        {filteredLines.map((line, index) => <div key={`${index}-${line}`} className="log-line">{line || " "}</div>)}
      </div>
    </div>
  );
}
