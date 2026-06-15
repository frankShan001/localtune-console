import React, { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Box, Cpu, Database, GitCompare, ListChecks, Loader2, WandSparkles } from "lucide-react";
import { apiErrorMessage, apiGet, apiPost, formatNumber } from "../api.js";
import { InputField, OpenFolderButton, PageToolbar, Panel, PropertyList, SelectField, TabBar } from "../components/ui.jsx";

function carriedInferenceTarget() {
  try {
    return JSON.parse(sessionStorage.getItem("localtune.inferenceTarget") || "{}");
  } catch {
    return {};
  }
}

export function InferencePage({ t }) {
  const carriedTarget = useState(carriedInferenceTarget)[0];
  const [mode, setMode] = useState(carriedTarget.mode || "single");
  const [modelId, setModelId] = useState(carriedTarget.model_id || "");
  const [branch, setBranch] = useState(carriedTarget.branch || "bnb4");
  const [adapter, setAdapter] = useState(carriedTarget.adapter || "");
  const [datasetProfile, setDatasetProfile] = useState("");
  const [batchRole, setBatchRole] = useState("test");
  const [batchLimit, setBatchLimit] = useState(5);
  const config = useQuery({ queryKey: ["config"], queryFn: () => apiGet("/api/config") });
  const target = useQuery({
    queryKey: ["inference-base-model", modelId, branch],
    queryFn: () => apiGet(`/api/inference/base-model?model_id=${encodeURIComponent(modelId)}&branch=${encodeURIComponent(branch)}`),
    enabled: Boolean(modelId),
  });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: () => apiGet("/api/artifacts") });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => apiGet("/api/datasets") });
  const adapters = (artifacts.data?.items || []).filter((item) => item.is_adapter && !item.archived && item.adapter_check?.ok !== false);
  const profiles = datasets.data?.profiles || [];
  const [prompt, setPrompt] = useState("请用目标作家的口吻改写这段话：今天的天气很好，我出门散步。");
  const [params, setParams] = useState({
    system_prompt: "",
    stop_words: "",
    max_input_tokens: 4096,
    max_new_tokens: 256,
    temperature: 0.7,
    top_p: 0.9,
  });
  const run = useMutation({
    mutationFn: () => {
      const payload = {
        branch,
        model_id: modelId,
        base_model: target.data?.base_model,
        adapter,
        prompt,
        compare: mode === "compare" || mode === "batch",
        dataset_profile: datasetProfile,
        role: batchRole,
        limit: Number(batchLimit),
        ...params,
      };
      return apiPost(mode === "batch" ? "/api/inference/batch" : "/api/inference/run", payload);
    },
  });

  useEffect(() => {
    if (!carriedTarget.branch && config.data?.active_branch) setBranch(config.data.active_branch);
  }, [carriedTarget.branch, config.data?.active_branch]);

  useEffect(() => {
    sessionStorage.removeItem("localtune.inferenceTarget");
  }, []);

  const selectedProfile = profiles.find((profile) => profile.id === datasetProfile);
  const batchReady = mode !== "batch" || Boolean(datasetProfile && selectedProfile?.[batchRole]?.exists);
  const canRun = Boolean(modelId && target.data?.base_model_exists && adapter && batchReady && !run.isPending);

  return (
    <div className="inference-workspace stack">
      <PageToolbar
        icon={WandSparkles}
        title={t("inference")}
        subtitle={t("inferenceHint")}
        stats={[
          { icon: Cpu, label: t("registeredModels"), value: config.data?.models?.length || 0 },
          { icon: Box, label: t("deployableAdapters"), value: adapters.length },
          { icon: Database, label: t("datasetProfiles"), value: profiles.length },
        ]}
        onRefresh={() => {
          config.refetch();
          if (modelId) target.refetch();
          artifacts.refetch();
          datasets.refetch();
        }}
        refreshing={config.isFetching || target.isFetching || artifacts.isFetching || datasets.isFetching}
        refreshLabel={t("refresh")}
      />
      <TabBar value={mode} onChange={setMode} items={[
        ["single", t("singleInference"), WandSparkles],
        ["compare", t("baseAdapterCompare"), GitCompare],
        ["batch", t("batchValidation"), ListChecks],
      ]} />

      <div className="two-column inference-layout">
        <Panel title={t("inferenceSetup")} subtitle={t("inferenceHint")}>
          <div className="form-grid">
            <SelectField label={t("baseModel")} value={modelId} onChange={setModelId} options={[
              ["", t("selectModel")],
              ...(config.data?.models || []).map((item) => [item.id, item.name || item.id]),
            ]} error={!modelId ? t("needSelectModel") : ""} />
            <SelectField label={t("branch")} value={branch} onChange={setBranch} options={(config.data?.branches || []).map((item) => [item.id, item.id])} />
            <SelectField label={t("adapter")} value={adapter} onChange={(value) => {
              setAdapter(value);
              const selected = adapters.find((item) => item.path === value);
              if (selected?.branch) setBranch(selected.branch);
              if (selected?.manifest?.model?.id) setModelId(selected.manifest.model.id);
            }} options={[
              ["", t("selectAdapter")],
              ...adapters.map((item) => [item.path, `${item.name} · ${item.branch}${item.best ? ` · ${t("bestArtifact")}` : ""}`]),
            ]} error={!adapter ? t("needSelectAdapter") : ""} />
          </div>
          <PropertyList rows={[
            [t("modelPath"), target.data?.base_model],
            [`${t("baseModel")} ${t("exists")}`, target.data?.base_model_exists ? t("yes") : t("no")],
            [t("adapter"), adapter],
            [`${t("adapter")} ${t("exists")}`, adapters.some((item) => item.path === adapter) ? t("yes") : t("no")],
          ]} />

          {mode !== "batch" && (
            <label className="field">
              <span>{t("prompt")}</span>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={7} />
            </label>
          )}

          {mode === "batch" && (
            <div className="form-grid">
              <SelectField label={t("datasetProfile")} value={datasetProfile} onChange={setDatasetProfile} options={[
                ["", t("selectDatasetProfile")],
                ...profiles.map((profile) => [profile.id, profile.name || profile.id]),
              ]} error={!datasetProfile ? t("needSelectDataset") : ""} />
              <SelectField label={t("datasetSplit")} value={batchRole} onChange={setBatchRole} options={[
                ["test", t("testData")],
                ["val", t("valData")],
                ["train", t("trainData")],
              ]} error={datasetProfile && !selectedProfile?.[batchRole]?.exists ? t("missingSplitHint") : ""} />
              <InputField label={t("sampleCount")} value={batchLimit} onChange={setBatchLimit} />
            </div>
          )}

          <label className="field">
            <span>{t("systemPrompt")}</span>
            <textarea value={params.system_prompt} onChange={(event) => setParams({ ...params, system_prompt: event.target.value })} rows={3} />
          </label>
          <div className="form-grid compact">
            <InputField label={t("maxInputTokens")} value={params.max_input_tokens} onChange={(max_input_tokens) => setParams({ ...params, max_input_tokens })} />
            <InputField label={t("maxNewTokens")} value={params.max_new_tokens} onChange={(max_new_tokens) => setParams({ ...params, max_new_tokens })} />
            <InputField label={t("temperature")} value={params.temperature} onChange={(temperature) => setParams({ ...params, temperature })} />
            <InputField label={t("topP")} value={params.top_p} onChange={(top_p) => setParams({ ...params, top_p })} />
            <InputField label={t("stopWords")} value={params.stop_words} onChange={(stop_words) => setParams({ ...params, stop_words })} placeholder={t("stopWordsHint")} />
          </div>
        </Panel>

        <Panel
          title={mode === "batch" ? t("evaluationReport") : t("result")}
          subtitle={run.data?.log_file || apiErrorMessage(run.error, t)}
          actions={(
            <>
              {run.data?.report_markdown && <OpenFolderButton path={run.data.report_markdown} label={t("openReport")} />}
              <button className="primary-button" disabled={!canRun} onClick={() => run.mutate()}>
                {run.isPending ? <Loader2 className="spin" size={17} /> : mode === "single" ? <WandSparkles size={17} /> : <GitCompare size={17} />}
                {mode === "batch" ? t("runBatchValidation") : t("runInference")}
              </button>
            </>
          )}
        >
          <InferenceResult mode={mode} data={run.data} error={run.error} t={t} />
        </Panel>
      </div>
    </div>
  );
}

function InferenceResult({ mode, data, error, t }) {
  if (error) return <pre className="result-box">{apiErrorMessage(error, t)}</pre>;
  if (!data) return <pre className="result-box">{t("waitingInferenceOutput")}</pre>;
  if (mode === "compare") {
    return (
      <div className="comparison-results">
        <article>
          <h3>{t("baseModelOutput")}</h3>
          <pre className="result-box">{data.base_response || "-"}</pre>
        </article>
        <article>
          <h3>{t("adapterOutput")}</h3>
          <pre className="result-box">{data.adapter_response || "-"}</pre>
        </article>
      </div>
    );
  }
  if (mode === "batch") {
    return (
      <div className="batch-results">
        <PropertyList rows={[
          [t("runId"), data.run_id],
          [t("sampleCount"), data.results?.length || 0],
          [t("elapsed"), `${formatNumber(data.elapsed_seconds, 1)} s`],
          [t("jsonReport"), data.report_json],
          [t("markdownReport"), data.report_markdown],
        ]} />
        {(data.results || []).map((item, index) => (
          <article className="batch-result" key={item.id || index}>
            <header><strong>{item.id || `#${index + 1}`}</strong></header>
            <p><b>{t("prompt")}:</b> {item.prompt}</p>
            <p><b>{t("expectedOutput")}:</b> {item.expected || "-"}</p>
            <div className="comparison-results compact">
              <section><h3>{t("baseModelOutput")}</h3><pre>{item.base_response || "-"}</pre></section>
              <section><h3>{t("adapterOutput")}</h3><pre>{item.adapter_response || "-"}</pre></section>
            </div>
          </article>
        ))}
      </div>
    );
  }
  return <pre className="result-box">{data.response || data.text || data.error || "-"}</pre>;
}
