import React, { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, Copy, Loader2, RefreshCw, WandSparkles } from "lucide-react";
import { apiErrorMessage, apiPost, formatBytes } from "../api.js";
import { ConfirmDialog, EmptyState, InputField, Panel, PropertyList, SelectField, ValidationBadge } from "../components/ui.jsx";

async function copyTextWithFallback(text) {
  try {
    await navigator.clipboard?.writeText(text);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      document.body.removeChild(textarea);
    }
  }
}

export function DeriveCorpusPanel({ t, files }) {
  const [payload, setPayload] = useState({ source: "", output: "data/processed/derived_sample.jsonl", mode: "head", limit: 20 });
  const [overwriteCandidate, setOverwriteCandidate] = useState(null);
  const derive = useMutation({ mutationFn: (requestPayload) => apiPost("/api/corpus/derive", requestPayload) });
  const selectedSource = files.find((file) => file.path === payload.source);

  useEffect(() => {
    if (!payload.source && files.length) setPayload((current) => ({ ...current, source: files[0].path }));
  }, [files, payload.source]);

  async function runDerive() {
    try {
      await derive.mutateAsync(payload);
    } catch (error) {
      if (String(error.message || "").includes("already exists")) {
        setOverwriteCandidate(payload);
      }
    }
  }

  return (
    <Panel title={t("deriveTrainableCorpus")} subtitle={t("deriveHint")}>
      <div className="derive-note">
        <strong>{t("deriveBoundaryTitle")}</strong>
        <span>{t("deriveBoundaryHint")}</span>
      </div>
      {!files.length && <EmptyState text={t("noTrainableCorpus")} />}
      <div className="form-grid">
        <SelectField label={t("sourceDataset")} value={payload.source} onChange={(source) => setPayload({ ...payload, source })} options={files.map((file) => [file.path, `${file.name} · ${file.rows ?? "-"} ${t("rowsUnit")}`])} />
        <InputField label={t("output")} value={payload.output} onChange={(output) => setPayload({ ...payload, output })} />
        <SelectField label={t("deriveMode")} value={payload.mode} onChange={(mode) => setPayload({ ...payload, mode })} options={[["head", t("head")], ["tail", t("tail")], ["sample", t("sample")], ["copy", t("copy")]]} />
        <InputField label={t("limit")} value={payload.limit} onChange={(limit) => setPayload({ ...payload, limit })} />
      </div>
      {selectedSource && <PropertyList rows={[
        [t("sourceDataset"), selectedSource.path],
        [t("sampleRows"), selectedSource.rows],
        [t("datasetFormat"), selectedSource.inferred_format],
        [t("size"), formatBytes(selectedSource.size_bytes)],
      ]} />}
      <button className="primary-button" disabled={!files.length || derive.isPending} onClick={runDerive}>
        {derive.isPending ? <Loader2 className="spin" size={17} /> : <WandSparkles size={17} />}
        {t("generateDerivedFile")}
      </button>
      {derive.data && <ValidationBadge result={derive.data} t={t} />}
      {derive.error && <div className="error-text">{apiErrorMessage(derive.error, t)}</div>}
      <ConfirmDialog
        open={Boolean(overwriteCandidate)}
        title={t("overwriteDerivedFileTitle")}
        message={t("overwriteDerivedFileConfirm")}
        confirmLabel={t("confirmOverwrite")}
        cancelLabel={t("cancel")}
        pending={derive.isPending}
        onCancel={() => setOverwriteCandidate(null)}
        onConfirm={() => {
          if (!overwriteCandidate) return;
          derive.mutate({ ...overwriteCandidate, overwrite: true });
          setOverwriteCandidate(null);
        }}
      />
    </Panel>
  );
}

export function DatasetProfilesPanel({ t, profiles = [] }) {
  const [validationOverrides, setValidationOverrides] = useState({});
  const validate = useMutation({
    mutationFn: (profileId) => apiPost("/api/datasets/validate", { profile: profileId, min_rows: 1 }),
    onSuccess: (result, profileId) => setValidationOverrides((current) => ({ ...current, [profileId]: result })),
  });

  return (
    <Panel title={t("datasetProfiles")} subtitle={t("datasetProfilesHint")}>
      <div className="profile-list">
        {profiles.map((profile) => {
          const validation = validationOverrides[profile.id] || profile.validation || {};
          const validating = validate.isPending && validate.variables === profile.id;
          return (
            <article className="profile-card" key={profile.id}>
              <header>
                <div>
                  <h3>{profile.name || profile.id}</h3>
                  <p>{profile.description || profile.id}</p>
                </div>
                <span className={validation.ok ? "status-pill completed" : "status-pill failed"}>
                  {validation.ok ? t("valid") : t("invalid")}
                </span>
              </header>
              <DatasetProfileSummary t={t} profile={profile} validation={validation} />
              <div className="profile-actions">
                <button className="secondary-button" disabled={validating} onClick={() => validate.mutate(profile.id)}>
                  {validating ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  {t("revalidate")}
                </button>
              </div>
              {validate.error && <div className="error-text">{apiErrorMessage(validate.error, t)}</div>}
            </article>
          );
        })}
        {!profiles.length && <EmptyState text={t("noData")} />}
      </div>
    </Panel>
  );
}

export function DatasetProfileSummary({ t, profile, validation }) {
  if (!profile) return <EmptyState text={t("noData")} />;
  const currentValidation = validation || profile.validation || {};
  return (
    <>
      <div className="profile-meta">
        <span>{t("taskType")}: {profile.task_type || "-"}</span>
        <span>{t("datasetFormat")}: {profile.format || "-"}</span>
        <span>{t("sampleRows")}: {profile.total_rows ?? "-"}</span>
        <span>{t("size")}: {formatBytes(profile.total_size_bytes)}</span>
      </div>
      <div className="profile-files">
        {["train", "val", "test"].map((role) => (
          <div className="profile-file" key={role}>
            <strong>{t(`${role}Data`)}</strong>
            <span title={profile[role]?.path || ""}>{profile[role]?.path || "-"}</span>
            <small>{profile[role]?.exists ? `${profile[role]?.rows ?? "-"} ${t("rowsUnit")} · ${formatBytes(profile[role]?.size_bytes)}` : t("missing")}</small>
          </div>
        ))}
      </div>
      <div className="profile-validation">
        <CheckCircle2 size={15} />
        <span>{currentValidation.ok ? t("validationPassed") : t("validationFailed")}</span>
        <small>
          {t("validationSummary", {
            errors: currentValidation.error_count ?? 0,
            warnings: currentValidation.warning_count ?? 0,
          })}
        </small>
      </div>
    </>
  );
}

export function CorpusGuide({ t }) {
  const [selectedType, setSelectedType] = useState("instruction");
  const [copied, setCopied] = useState(false);
  const examples = [
    {
      id: "instruction",
      title: t("instructionType"),
      fields: "instruction / input / output",
      description: t("instructionCorpusHint"),
      path: "examples/corpus/instruction.jsonl",
      example: {
        task_type: "instruction",
        instruction: "请给出三条减少无效会议的建议。",
        input: "面向一个每周会议较多的产品团队。",
        output: "提前发送议程；能异步同步的内容改用文档；为每场会议设置明确的决策目标。",
        metadata: { id: "instruction-0001", split: "train" },
      },
    },
    {
      id: "rewrite",
      title: t("rewriteType"),
      fields: "source / target",
      description: t("rewriteCorpusHint"),
      path: "examples/corpus/rewrite.jsonl",
      example: {
        task_type: "rewrite",
        source: "那座旧院已经空了很多年，台阶上都是灰，但屋子仍然宽敞明亮。",
        target: "旧院空置经年，阶前积尘，推门望去，屋舍却依旧宽朗。",
        metadata: { id: "rewrite-0001", style: "简洁文学", split: "train" },
      },
    },
    {
      id: "chat",
      title: t("chatType"),
      fields: "messages[]",
      description: t("chatCorpusHint"),
      path: "examples/corpus/chat.jsonl",
      example: {
        task_type: "chat",
        messages: [
          { role: "system", content: "你是一名回答简洁的助手。" },
          { role: "user", content: "为什么训练前要检查语料？" },
          { role: "assistant", content: "可以提前发现格式错误、字段缺失和空输出，避免浪费训练时间。" },
        ],
        metadata: { id: "chat-0001", split: "train" },
      },
    },
    {
      id: "dpo",
      title: t("dpoType"),
      fields: "prompt / chosen / rejected",
      description: t("dpoCorpusHint"),
      path: "examples/corpus/dpo.jsonl",
      example: {
        task_type: "dpo",
        prompt: "请简要说明训练前检查语料的意义。",
        chosen: "语料检查可以提前发现结构和内容问题，避免无效训练。",
        rejected: "因为检查一下通常会比较好。",
        metadata: { id: "dpo-0001", split: "train" },
      },
    },
  ];
  const selected = examples.find((item) => item.id === selectedType) || examples[0];
  const exampleText = JSON.stringify(selected.example, null, 2);
  const exampleJsonl = JSON.stringify(selected.example);

  async function copyExample() {
    const ok = await copyTextWithFallback(exampleJsonl);
    if (!ok) return;
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <Panel title={t("localTuneCorpusFormat")} subtitle={t("corpusFormatGuideHint")}>
      <div className="corpus-guide-layout">
        <div className="guide-grid">
          {examples.map((item) => (
            <button
              className={selected.id === item.id ? "guide-card active" : "guide-card"}
              key={item.id}
              type="button"
              aria-pressed={selected.id === item.id}
              onClick={() => setSelectedType(item.id)}
            >
              <h3>{item.title}</h3>
              <code>{item.fields}</code>
              <p>{item.description}</p>
            </button>
          ))}
        </div>
        <section className="corpus-example">
          <header>
            <div>
              <h3>{selected.title} {t("corpusExample")}</h3>
              <span>{selected.path}</span>
            </div>
            <button className="secondary-button" type="button" onClick={copyExample}>
              <Copy size={15} /> {copied ? t("copied") : t("copyJsonlLine")}
            </button>
          </header>
          <p className="jsonl-copy-hint">{t("jsonlCopyHint")}</p>
          <pre className="result-box">{exampleText}</pre>
        </section>
      </div>
    </Panel>
  );
}

export function inferCommonProperties(samples, file) {
  const first = samples[0] || {};
  const taskTypes = new Set(samples.map((sample) => sample.task_type).filter(Boolean));
  const formats = new Set(samples.map((sample) => sample.format).filter(Boolean));
  return {
    taskType: taskTypes.size === 1 ? [...taskTypes][0] : first.task_type || file?.task_type || file?.extension || "-",
    format: formats.size === 1 ? [...formats][0] : first.format || file?.inferred_format || file?.extension || "-",
    trainable: samples.length ? samples.every((sample) => sample.trainable !== false) : Boolean(file?.trainable),
  };
}
