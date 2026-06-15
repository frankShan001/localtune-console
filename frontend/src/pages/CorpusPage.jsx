import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, BookOpen, CheckCircle2, ChevronLeft, ChevronRight, Copy, Database, FileJson, FilePlus2, FileText, Info, ListFilter, Loader2, Pencil, Plus, Scissors, Search, Trash2, WandSparkles, X } from "lucide-react";
import { apiDelete, apiErrorMessage, apiGet, apiPost, apiPut, formatBytes, formatDate, shortPath } from "../api.js";
import {
  ContextDrawer,
  ConfirmDialog,
  EmptyState,
  InputField,
  OpenFolderButton,
  PageToolbar,
  Panel,
  PropertyList,
  SampleCard,
  SelectControl,
  SelectField,
  TabBar,
  ValidationBadge,
} from "../components/ui.jsx";
import { CorpusGuide, DeriveCorpusPanel } from "./corpusSupport.jsx";

function normalizeDatasetFile(file) {
  return {
    ...file,
    extension: file.extension || "",
    inferred_format: file.inferred_format || file.format || file.extension || "-",
    task_type: file.task_type || "-",
    trainable: Boolean(file.trainable),
  };
}

function profileFile(profile, role) {
  return profile?.[role] || null;
}

function fileReady(file) {
  return Boolean(file?.exists);
}

function trainingDatasetReadiness(profile, t) {
  const trainReady = fileReady(profileFile(profile, "train"));
  const valReady = fileReady(profileFile(profile, "val"));
  const testReady = fileReady(profileFile(profile, "test"));
  const validationFailed = profile?.validation?.ok === false;
  if (!trainReady) {
    return { state: "warn", ready: false, label: t("notReadyToTrain"), hint: t("trainingDatasetMissingTrainHint") };
  }
  if (validationFailed) {
    return { state: "warn", ready: false, label: t("notReadyToTrain"), hint: t("trainingDatasetValidationFailedHint") };
  }
  if (!valReady && !testReady) {
    return { state: "partial", ready: true, label: t("readyForTrialRun"), hint: t("trainingDatasetTrainOnlyHint") };
  }
  if (!valReady) {
    return { state: "partial", ready: true, label: t("readyForTrialRun"), hint: t("trainingDatasetMissingValHint") };
  }
  if (!testReady) {
    return { state: "partial", ready: true, label: t("readyForTrialRun"), hint: t("trainingDatasetMissingTestHint") };
  }
  return { state: "ok", ready: true, label: t("readyToTrain"), hint: t("trainingDatasetAllSplitsReadyHint") };
}

function missingSplitRoles(profile) {
  return ["val", "test"].filter((role) => !fileReady(profileFile(profile, role)));
}

function roleLabel(role, t) {
  if (role === "train") return t("trainData");
  if (role === "val") return t("valData");
  return t("testData");
}

export function CorpusPage({ t }) {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState("existing");
  const [libraryMode, setLibraryMode] = useState("trainable");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [selectedRole, setSelectedRole] = useState("train");
  const [selectedMaterialPath, setSelectedMaterialPath] = useState("");
  const [previewLimit, setPreviewLimit] = useState(20);
  const [previewQuery, setPreviewQuery] = useState("");
  const [previewPage, setPreviewPage] = useState(0);
  const [checkResult, setCheckResult] = useState(null);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [profileEditor, setProfileEditor] = useState(null);
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const registry = useQuery({ queryKey: ["dataset-registry"], queryFn: () => apiGet("/api/datasets/registry") });
  const library = useQuery({ queryKey: ["corpus-library"], queryFn: () => apiGet("/api/corpus/library"), enabled: !registry.data });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => apiGet("/api/datasets") });
  const profiles = datasets.data?.profiles || [];
  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || profiles[0];
  const scannedFiles = (registry.data?.files || library.data?.files || []).map(normalizeDatasetFile);
  const files = scannedFiles.filter((file) => file.trainable);
  const materialFiles = (registry.data?.materials?.length ? registry.data.materials : scannedFiles.filter((file) => !file.trainable)).map(normalizeDatasetFile);
  const selectedSplitFile = profileFile(selectedProfile, selectedRole);
  const activePath = libraryMode === "materials"
    ? selectedMaterialPath
    : fileReady(selectedSplitFile) ? selectedSplitFile.path : "";
  const materialFile = materialFiles.find((file) => file.path === selectedMaterialPath);
  const selectedFile = libraryMode === "materials" ? materialFile : selectedSplitFile;
  const ActiveFileIcon = libraryMode === "materials" ? FileText : FileJson;
  const preview = useQuery({
    queryKey: ["corpus-preview", activePath, previewLimit, previewPage, previewQuery],
    queryFn: () => apiGet(`/api/corpus/preview?path=${encodeURIComponent(activePath)}&limit=${previewLimit}&offset=${previewPage * previewLimit}&query=${encodeURIComponent(previewQuery)}`),
    enabled: Boolean(activePath),
  });
  const missingRoles = missingSplitRoles(selectedProfile);

  useEffect(() => {
    if (profiles.length && !profiles.some((profile) => profile.id === selectedProfileId)) {
      setSelectedProfileId(profiles[0].id);
    }
  }, [profiles, selectedProfileId]);

  useEffect(() => {
    if (!selectedMaterialPath && materialFiles.length) {
      setSelectedMaterialPath(materialFiles[0].path);
    }
  }, [materialFiles, selectedMaterialPath]);

  useEffect(() => {
    setCheckResult(null);
    setPreviewPage(0);
    setPreviewQuery("");
  }, [libraryMode, selectedProfileId, selectedRole]);

  useEffect(() => {
    if (!checkResult || checkResult.pending) return undefined;
    const timer = window.setTimeout(() => setCheckResult(null), 4200);
    return () => window.clearTimeout(timer);
  }, [checkResult]);

  useEffect(() => {
    if (!deleteCandidate) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") setDeleteCandidate(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteCandidate]);

  useEffect(() => {
    if (libraryMode !== "trainable") return;
    if (!fileReady(profileFile(selectedProfile, selectedRole)) && fileReady(profileFile(selectedProfile, "train"))) {
      setSelectedRole("train");
    }
  }, [libraryMode, selectedProfile, selectedRole]);

  const materialRows = useMemo(() => [
    [t("materialFile"), selectedFile?.name],
    [t("filePath"), selectedFile?.path],
    [t("folder"), selectedFile?.folder],
    [t("fileType"), selectedFile?.extension || selectedFile?.inferred_format],
    [t("size"), formatBytes(selectedFile?.size_bytes)],
    [t("updated"), formatDate(selectedFile?.updated)],
  ], [selectedFile, t]);

  const profileRows = useMemo(() => [
    [t("datasetProfile"), selectedProfile?.name || selectedProfile?.id],
    [t("taskType"), selectedProfile?.task_type],
    [t("datasetFormat"), selectedProfile?.format],
    [t("sampleRows"), selectedProfile?.total_rows],
    [t("size"), formatBytes(selectedProfile?.total_size_bytes)],
    [t("validation"), selectedProfile?.validation?.ok ? t("validationPassed") : t("validationFailed")],
  ], [selectedProfile, t]);

  const toolbarTitle = libraryMode === "materials"
    ? selectedFile?.name || t("rawMaterials")
    : selectedProfile?.name || t("selectDatasetProfile");
  const toolbarSubtitle = libraryMode === "materials"
    ? shortPath(selectedMaterialPath) || t("materialHint")
    : selectedProfile?.description || t("trainingDatasetsHint");
  const toolbarHint = libraryMode === "materials"
    ? t("materialHint")
    : t("trainingDatasetsHint");

  useEffect(() => {
    if (!isPickerOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") setIsPickerOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isPickerOpen]);

  async function runCheck() {
    if (libraryMode !== "trainable" || !activePath) return;
    setCheckResult({ pending: true });
    try {
      const result = await apiPost("/api/corpus/check", {
        path: activePath,
        task_type: selectedProfile?.task_type || registry.data?.task_type || "chatml",
        format: selectedProfile?.format || registry.data?.format || "chatml_source",
      });
      setCheckResult(result);
    } catch (error) {
      setCheckResult({ ok: false, error: apiErrorMessage(error, t) });
    }
  }

  function refreshDatasets() {
    queryClient.invalidateQueries({ queryKey: ["datasets"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-registry"] });
    queryClient.invalidateQueries({ queryKey: ["corpus-library"] });
    setCheckResult(null);
  }

  const saveProfile = useMutation({
    mutationFn: ({ mode, originalId, payload }) => {
      if (mode === "edit") {
        return apiPut(`/api/datasets/profiles/${encodeURIComponent(originalId)}`, payload);
      }
      if (mode === "copy") {
        return apiPost(`/api/datasets/profiles/${encodeURIComponent(originalId)}/copy`, payload);
      }
      return apiPost("/api/datasets/profiles", payload);
    },
    onSuccess: (data) => {
      setSelectedProfileId(data.profile);
      setProfileEditor(null);
      refreshDatasets();
    },
  });
  const deleteProfile = useMutation({
    mutationFn: (profileId) => apiDelete(`/api/datasets/profiles/${encodeURIComponent(profileId)}`),
    onSuccess: () => {
      setSelectedProfileId("");
      refreshDatasets();
    },
  });
  const importCorpus = useMutation({
    mutationFn: () => apiPost("/api/corpus/import", {}),
    onSuccess: (data) => {
      if (data?.cancelled) return;
      setLibraryMode("trainable");
      setTab("existing");
      setSelectedProfileId(data.profile || "");
      setSelectedRole("train");
      setCheckResult(data.validation || null);
      refreshDatasets();
    },
  });

  function requestDeleteProfile() {
    if (!selectedProfile) return;
    setDeleteCandidate(selectedProfile);
  }

  return (
    <div className="stack">
      <PageToolbar
        icon={Database}
        title={t("corpus")}
        subtitle={t("corpusHint")}
        stats={[
          { icon: FileJson, label: t("datasetProfiles"), value: profiles.length },
          { icon: CheckCircle2, label: t("trainableCorpusFiles"), value: files.length },
          { icon: FileText, label: t("rawMaterials"), value: materialFiles.length },
        ]}
        actions={(
          <button className="primary-button" disabled={importCorpus.isPending} onClick={() => importCorpus.mutate()}>
            {importCorpus.isPending ? <Loader2 className="spin" size={16} /> : <FilePlus2 size={16} />}
            {t("addTrainingDataset")}
          </button>
        )}
        onRefresh={refreshDatasets}
        refreshing={registry.isFetching || library.isFetching || datasets.isFetching}
        refreshLabel={t("refresh")}
      />
      <TabBar value={tab} onChange={setTab} items={[
        ["existing", t("existingCorpus"), FileJson],
        ["guide", t("formatGuide"), BookOpen, "amber"],
        ["derive", t("deriveTrainableCorpus"), WandSparkles],
      ]} />
      {tab === "existing" && (
        <div className="corpus-manager stack">
          <CorpusDataManager
            profiles={profiles}
            files={files}
            materialFiles={materialFiles}
            libraryMode={libraryMode}
            setLibraryMode={setLibraryMode}
            selectedProfile={selectedProfile}
            selectedMaterialPath={selectedMaterialPath}
            selectedFile={selectedFile}
            selectedRole={selectedRole}
            setSelectedRole={setSelectedRole}
            activePath={activePath}
            selectedSplitFile={selectedSplitFile}
            preview={preview}
            previewQuery={previewQuery}
            setPreviewQuery={setPreviewQuery}
            previewLimit={previewLimit}
            setPreviewLimit={setPreviewLimit}
            previewPage={previewPage}
            setPreviewPage={setPreviewPage}
            checkResult={checkResult}
            runCheck={runCheck}
            profileRows={profileRows}
            materialRows={materialRows}
            missingRoles={missingRoles}
            refreshDatasets={refreshDatasets}
            setSelectedProfileId={setSelectedProfileId}
            setSelectedMaterialPath={setSelectedMaterialPath}
            setCheckResult={setCheckResult}
            setProfileEditor={setProfileEditor}
            requestDeleteProfile={requestDeleteProfile}
            deletePending={deleteProfile.isPending}
            importResult={importCorpus.data}
            importError={importCorpus.error}
            t={t}
          />
          <section className="corpus-toolbar">
            <div className="corpus-toolbar-main">
              <div className="corpus-toolbar-copy">
                <strong>{toolbarTitle}</strong>
                <span>{toolbarSubtitle}</span>
              </div>
              <div className="corpus-toolbar-actions">
                {libraryMode === "trainable" && (
                  <>
                    <button className="primary-button" disabled={importCorpus.isPending} onClick={() => importCorpus.mutate()}>
                      {importCorpus.isPending ? <Loader2 className="spin" size={16} /> : <FilePlus2 size={16} />}
                      {t("importTrainingCorpus")}
                    </button>
                    <button className="secondary-button" onClick={() => setProfileEditor({ mode: "create" })}>
                      <Plus size={16} /> {t("createDatasetProfile")}
                    </button>
                    <button className="secondary-button" disabled={!selectedProfile} onClick={() => setProfileEditor({ mode: "edit", profile: selectedProfile })}>
                      <Pencil size={16} /> {t("edit")}
                    </button>
                    <button className="secondary-button" disabled={!selectedProfile} onClick={() => setProfileEditor({ mode: "copy", profile: selectedProfile })}>
                      <Copy size={16} /> {t("copyDatasetProfile")}
                    </button>
                    <button className="danger-button" disabled={!selectedProfile || deleteProfile.isPending} onClick={requestDeleteProfile}>
                      <Trash2 size={16} /> {t("delete")}
                    </button>
                  </>
                )}
                <button className="secondary-button" onClick={() => setIsPickerOpen(true)}>
                  <ListFilter size={16} />
                  {libraryMode === "materials" ? t("selectMaterial") : t("selectTrainingDataset")}
                </button>
              </div>
            </div>
            <div className="corpus-mode-row">
              <div className="corpus-segment" role="tablist" aria-label={t("corpusLibrary")}>
                <button className={libraryMode === "trainable" ? "active" : ""} onClick={() => setLibraryMode("trainable")}>
                  <FileJson size={15} />
                  {t("trainingDatasets")}
                  <span>{profiles.length}</span>
                </button>
                <button className={libraryMode === "materials" ? "active" : ""} onClick={() => setLibraryMode("materials")}>
                  <FileText size={15} />
                  {t("rawMaterials")}
                  <span>{materialFiles.length}</span>
                </button>
              </div>
              <p>{toolbarHint}</p>
            </div>
            {importCorpus.data?.file && (
              <div className="success-text corpus-import-status">
                {t("importTrainingCorpusDone", { name: importCorpus.data.file.name })}
              </div>
            )}
            {importCorpus.error && (
              <div className="error-text corpus-import-status">{apiErrorMessage(importCorpus.error, t)}</div>
            )}
          </section>

          {isPickerOpen && (
            <div className="artifact-picker-backdrop" onMouseDown={() => setIsPickerOpen(false)}>
              <aside className="artifact-picker corpus-picker" onMouseDown={(event) => event.stopPropagation()}>
                <header className="artifact-picker-head">
                  <div>
                    <h2>{libraryMode === "materials" ? t("rawMaterials") : t("trainingDatasets")}</h2>
                    <p>{libraryMode === "materials" ? `${materialFiles.length} ${t("materialFiles")}` : `${profiles.length} ${t("datasetProfiles")}`}</p>
                  </div>
                  <button className="secondary-button icon-only" title={t("close")} onClick={() => setIsPickerOpen(false)}>
                    <X size={18} />
                  </button>
                </header>
                <div className="file-list">
                  {libraryMode === "trainable" && profiles.map((profile) => (
                    <button key={profile.id} className={profile.id === selectedProfile?.id ? "file-row active" : "file-row"} onClick={() => {
                      setSelectedProfileId(profile.id);
                      setSelectedRole("train");
                      setCheckResult(null);
                      setIsPickerOpen(false);
                    }}>
                      <FileJson size={16} />
                      <span>
                        <strong>{profile.name || profile.id}</strong>
                        <small>{profile.total_rows ?? "-"} {t("rowsUnit")} · {profile.train?.path || "-"} · {profile.validation?.ok ? t("valid") : t("invalid")}</small>
                      </span>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                  {libraryMode === "materials" && materialFiles.map((file) => (
                    <button key={file.path} className={file.path === selectedMaterialPath ? "file-row active" : "file-row"} onClick={() => {
                      setSelectedMaterialPath(file.path);
                      setCheckResult(null);
                      setIsPickerOpen(false);
                    }}>
                      <ActiveFileIcon size={16} />
                      <span>
                        <strong>{file.name}</strong>
                        <small>{file.folder} · {file.extension || file.inferred_format} · {formatBytes(file.size_bytes)} · {formatDate(file.updated)}</small>
                      </span>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                  {libraryMode === "trainable" && !profiles.length && <EmptyState text={t("noData")} />}
                  {libraryMode === "materials" && !materialFiles.length && <EmptyState text={t("noData")} />}
                </div>
              </aside>
            </div>
          )}

          {libraryMode === "trainable" && (
            <div className="corpus-workspace">
              <div className="stack">
                <Panel title={t("datasetProfile")} subtitle={selectedProfile?.id || ""}>
                  {selectedProfile ? <PropertyList rows={profileRows} /> : <EmptyState text={t("selectDatasetProfileToInspect")} />}
                </Panel>
                <Panel title={t("datasetComposition")} subtitle={t("datasetCompositionHint")}>
                  {selectedProfile ? (
                    <div className="split-card-list">
                      {["train", "val", "test"].map((role) => (
                        <DatasetSplitCard
                          key={role}
                          role={role}
                          file={profileFile(selectedProfile, role)}
                          active={selectedRole === role}
                          onSelect={() => setSelectedRole(role)}
                          t={t}
                        />
                      ))}
                    </div>
                  ) : (
                    <EmptyState text={t("selectDatasetProfileToInspect")} />
                  )}
                </Panel>
                {selectedProfile && missingRoles.length > 0 && (
                  <ProfileSplitPanel profile={selectedProfile} missingRoles={missingRoles} onDone={refreshDatasets} t={t} />
                )}
              </div>

              <Panel
                title={`${roleLabel(selectedRole, t)} ${t("samplePreview")}`}
                subtitle={activePath ? selectedSplitFile?.path : t("missingSplitHint")}
                actions={(
                  <>
                    <label className="compact-search">
                      <Search size={15} />
                      <input
                        value={previewQuery}
                        placeholder={t("searchSamples")}
                        onChange={(event) => {
                          setPreviewQuery(event.target.value);
                          setPreviewPage(0);
                        }}
                      />
                    </label>
                    <SelectControl
                      value={String(previewLimit)}
                      onChange={(limit) => setPreviewLimit(Number(limit))}
                      ariaLabel={t("samplesUnit")}
                      options={[20, 50, 100].map((limit) => [String(limit), `${limit} ${t("samplesUnit")}`])}
                    />
                    <button className="secondary-button" disabled={!activePath} onClick={runCheck}>
                      {checkResult?.pending ? <Loader2 className="spin" size={16} /> : <CheckCircle2 size={16} />}
                      {t("formatCheck")}
                    </button>
                    <OpenFolderButton path={activePath} label={t("openFolder")} />
                  </>
                )}
              >
                {checkResult && !checkResult.pending && <ValidationBadge result={checkResult} t={t} />}
                <section className="reader">
                  <div className="reader-pagination">
                    <span>{t("sampleRange", {
                      start: preview.data?.pagination?.total ? previewPage * previewLimit + 1 : 0,
                      end: previewPage * previewLimit + (preview.data?.samples?.length || 0),
                      total: preview.data?.pagination?.total || 0,
                    })}</span>
                    <div>
                      <button className="secondary-button icon-only" title={t("previousPage")} disabled={!preview.data?.pagination?.has_previous} onClick={() => setPreviewPage((page) => Math.max(0, page - 1))}>
                        <ChevronLeft size={16} />
                      </button>
                      <button className="secondary-button icon-only" title={t("nextPage")} disabled={!preview.data?.pagination?.has_next} onClick={() => setPreviewPage((page) => page + 1)}>
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                  <div className="sample-list">
                    {preview.isLoading && <EmptyState text={t("loadingCorpusSamples")} />}
                    {!activePath && <EmptyState text={t("missingSplitHint")} />}
                    {activePath && !preview.isLoading && !preview.data?.samples?.length && <EmptyState text={t("noData")} />}
                    {activePath && preview.data?.samples?.map((sample) => (
                      <SampleCard key={sample.line} sample={sample} t={t} />
                    ))}
                  </div>
                </section>
              </Panel>
            </div>
          )}

          {libraryMode === "materials" && (
            <div className="corpus-workspace">
              <Panel
                title={t("materialInfo")}
                subtitle={t("materialPropertiesHint")}
                actions={<OpenFolderButton path={selectedMaterialPath} label={t("openFolder")} />}
              >
                <PropertyList rows={materialRows} />
              </Panel>
              <Panel title={t("contentPreview")} subtitle={t("materialPreviewHint")}>
                <section className="reader">
                  <div className="sample-list">
                    {preview.isLoading && <EmptyState text={t("loadingCorpusSamples")} />}
                    {!preview.isLoading && !preview.data?.samples?.length && <EmptyState text={t("noData")} />}
                    {preview.data?.samples?.length > 0 && <MaterialContentPreview samples={preview.data.samples} t={t} />}
                  </div>
                </section>
              </Panel>
            </div>
          )}
        </div>
      )}
      {tab === "guide" && <CorpusGuide t={t} />}
      {tab === "derive" && <DeriveCorpusPanel t={t} files={files} />}
      <ProfileEditorDrawer
        editor={profileEditor}
        files={files}
        saving={saveProfile.isPending}
        error={saveProfile.error}
        onClose={() => setProfileEditor(null)}
        onSave={(payload) => saveProfile.mutate({
          mode: profileEditor.mode,
          originalId: profileEditor.profile?.id,
          payload,
        })}
        t={t}
      />
      <ConfirmDialog
        open={Boolean(deleteCandidate)}
        title={t("deleteDatasetProfileTitle")}
        message={deleteCandidate ? t("deleteDatasetProfileConfirm", { name: deleteCandidate.name || deleteCandidate.id }) : ""}
        confirmLabel={t("confirmDelete")}
        cancelLabel={t("cancel")}
        pending={deleteProfile.isPending}
        onCancel={() => setDeleteCandidate(null)}
        onConfirm={() => {
          if (!deleteCandidate) return;
          deleteProfile.mutate(deleteCandidate.id);
          setDeleteCandidate(null);
        }}
      />
    </div>
  );
}

function CorpusDataManager({
  profiles,
  files,
  materialFiles,
  libraryMode,
  setLibraryMode,
  selectedProfile,
  selectedMaterialPath,
  selectedFile,
  selectedRole,
  setSelectedRole,
  activePath,
  selectedSplitFile,
  preview,
  previewQuery,
  setPreviewQuery,
  previewLimit,
  setPreviewLimit,
  previewPage,
  setPreviewPage,
  checkResult,
  runCheck,
  profileRows,
  materialRows,
  missingRoles,
  refreshDatasets,
  setSelectedProfileId,
  setSelectedMaterialPath,
  setCheckResult,
  setProfileEditor,
  requestDeleteProfile,
  deletePending,
  importResult,
  importError,
  t,
}) {
  const [assetQuery, setAssetQuery] = useState("");
  const normalizedAssetQuery = assetQuery.trim().toLowerCase();
  const filteredProfiles = normalizedAssetQuery
    ? profiles.filter((profile) => [
      profile.id,
      profile.name,
      profile.description,
      profile.task_type,
      profile.format,
      profile.train?.path,
      profile.val?.path,
      profile.test?.path,
    ].filter(Boolean).some((value) => String(value).toLowerCase().includes(normalizedAssetQuery)))
    : profiles;
  const filteredMaterials = normalizedAssetQuery
    ? materialFiles.filter((file) => [
      file.name,
      file.path,
      file.folder,
      file.extension,
      file.inferred_format,
    ].filter(Boolean).some((value) => String(value).toLowerCase().includes(normalizedAssetQuery)))
    : materialFiles;
  const selectedProfileVisible = !normalizedAssetQuery || filteredProfiles.some((profile) => profile.id === selectedProfile?.id);
  const selectedMaterialVisible = !normalizedAssetQuery || filteredMaterials.some((file) => file.path === selectedMaterialPath);
  const visibleProfile = selectedProfileVisible ? selectedProfile : null;
  const visibleMaterialPath = selectedMaterialVisible ? selectedMaterialPath : "";
  const visibleFile = libraryMode === "materials" ? (selectedMaterialVisible ? selectedFile : null) : selectedFile;
  const visibleActivePath = libraryMode === "materials"
    ? visibleMaterialPath
    : visibleProfile ? activePath : "";
  const visibleSplitFile = visibleProfile ? selectedSplitFile : null;
  const visibleMaterialRows = selectedMaterialVisible ? materialRows : [];
  const visibleMissingRoles = visibleProfile ? missingRoles : [];
  const noAssetMatch = normalizedAssetQuery && (
    libraryMode === "materials" ? !filteredMaterials.length : !filteredProfiles.length
  );

  return (
    <section className="corpus-data-manager-v7">
      <CorpusAssetRail
        profiles={filteredProfiles}
        totalProfiles={profiles.length}
        materialFiles={filteredMaterials}
        totalMaterialFiles={materialFiles.length}
        selectedProfile={visibleProfile}
        selectedMaterialPath={visibleMaterialPath}
        libraryMode={libraryMode}
        assetQuery={assetQuery}
        setAssetQuery={setAssetQuery}
        onSelectProfile={(profile) => {
          setLibraryMode("trainable");
          setSelectedProfileId(profile.id);
          setSelectedRole("train");
          setCheckResult(null);
        }}
        onSelectMaterial={(file) => {
          setLibraryMode("materials");
          setSelectedMaterialPath(file.path);
          setCheckResult(null);
        }}
        t={t}
      />

      {libraryMode === "trainable" ? (
        <main className="corpus-data-body">
          {noAssetMatch ? (
            <section className="corpus-no-match">
              <EmptyState text={t("noMatchingDatasetProfiles")} />
            </section>
          ) : (
            <>
              <section className="corpus-management-pane">
                <CorpusDatasetSummary profile={visibleProfile} t={t} />
                <DatasetSplitTable
                  profile={visibleProfile}
                  selectedRole={selectedRole}
                  setSelectedRole={setSelectedRole}
                  t={t}
                />
              </section>
              <CorpusInspector
                profile={visibleProfile}
                activePath={visibleActivePath}
                checkResult={checkResult}
                runCheck={runCheck}
                onEdit={() => visibleProfile && setProfileEditor({ mode: "edit", profile: visibleProfile })}
                onCopy={() => visibleProfile && setProfileEditor({ mode: "copy", profile: visibleProfile })}
                onCreate={() => setProfileEditor({ mode: "create" })}
                onDelete={requestDeleteProfile}
                deletePending={deletePending}
                t={t}
              />
              {visibleProfile && visibleMissingRoles.length > 0 && (
                <section className="corpus-split-helper">
                  <ProfileSplitPanel profile={visibleProfile} missingRoles={visibleMissingRoles} onDone={refreshDatasets} t={t} />
                </section>
              )}
              <CorpusSampleReader
                title={`${roleLabel(selectedRole, t)} ${t("samplePreview")}`}
                subtitle={visibleActivePath ? visibleSplitFile?.path : t("missingSplitHint")}
                activePath={visibleActivePath}
                preview={preview}
                previewQuery={previewQuery}
                setPreviewQuery={setPreviewQuery}
                previewLimit={previewLimit}
                setPreviewLimit={setPreviewLimit}
                previewPage={previewPage}
                setPreviewPage={setPreviewPage}
                checkResult={checkResult}
                t={t}
              />
            </>
          )}
        </main>
      ) : (
        <main className="corpus-data-body materials">
          <section className="corpus-management-pane">
            <div className="corpus-object-head">
              <span className="corpus-kicker">{t("rawMaterials")}</span>
              <h2>{visibleFile?.name || t("rawMaterials")}</h2>
              <p>{visibleMaterialPath || t("materialHint")}</p>
            </div>
            <PropertyList rows={visibleMaterialRows} />
          </section>
          <aside className="corpus-inspector">
            <h2>{t("materialInfo")}</h2>
            <p>{t("materialPropertiesHint")}</p>
            <OpenFolderButton path={visibleMaterialPath} label={t("openFolder")} />
          </aside>
          <section className="corpus-wide-reader">
            <div className="corpus-reader-head">
              <div>
                <span className="corpus-kicker">{t("contentPreview")}</span>
                <h2>{t("contentPreview")}</h2>
                <p>{t("materialPreviewHint")}</p>
              </div>
            </div>
            <div className="sample-list corpus-sample-list">
              {noAssetMatch && <EmptyState text={t("noMatchingMaterials")} />}
              {!noAssetMatch && preview.isLoading && <EmptyState text={t("loadingCorpusSamples")} />}
              {!noAssetMatch && !preview.isLoading && !preview.data?.samples?.length && <EmptyState text={t("noData")} />}
              {!noAssetMatch && preview.data?.samples?.length > 0 && <MaterialContentPreview samples={preview.data.samples} t={t} />}
            </div>
          </section>
        </main>
      )}

      {importResult?.file && (
        <div className="success-text corpus-import-status">
          {t("importTrainingCorpusDone", { name: importResult.file.name })}
        </div>
      )}
      {importError && (
        <div className="error-text corpus-import-status">{apiErrorMessage(importError, t)}</div>
      )}
    </section>
  );
}

function CorpusAssetRail({ profiles, totalProfiles, materialFiles, totalMaterialFiles, selectedProfile, selectedMaterialPath, libraryMode, assetQuery, setAssetQuery, onSelectProfile, onSelectMaterial, t }) {
  return (
    <aside className="corpus-asset-rail">
      <header>
        <span className="corpus-kicker">{t("trainingDatasets")}</span>
        <label className="compact-search corpus-asset-search">
          <Search size={15} />
          <input value={assetQuery} placeholder={t("searchDatasets")} onChange={(event) => setAssetQuery(event.target.value)} />
        </label>
      </header>
      <div className="corpus-asset-table-head">
        <span>{t("name")}</span>
        <span>{t("sampleRows")}</span>
        <span>{t("datasetStatus")}</span>
      </div>
      <div className="corpus-asset-list">
        {profiles.map((profile) => {
          const active = libraryMode === "trainable" && profile.id === selectedProfile?.id;
          const ready = Boolean(profile.validation?.ok && fileReady(profile.train));
          return (
            <button key={profile.id} className={active ? "corpus-asset-row active" : "corpus-asset-row"} onClick={() => onSelectProfile(profile)}>
              <span>
                <strong>{profile.name || profile.id}</strong>
                <small>{profile.task_type || "-"} · {profile.format || "-"}</small>
              </span>
              <span>{profile.total_rows ?? "-"}</span>
              <span className={ready ? "corpus-dot ok" : "corpus-dot warn"}>{ready ? t("trainable") : t("issue")}</span>
            </button>
          );
        })}
        {!profiles.length && <EmptyState text={totalProfiles ? t("noMatchingDatasetProfiles") : t("noDatasetProfiles")} />}
      </div>
      <div className="corpus-asset-materials">
        <span className="corpus-kicker">{t("rawMaterials")}</span>
        {materialFiles.map((file) => (
          <button key={file.path} className={libraryMode === "materials" && file.path === selectedMaterialPath ? "corpus-material-row active" : "corpus-material-row"} onClick={() => onSelectMaterial(file)}>
            <strong>{file.name}</strong>
            <small>{formatBytes(file.size_bytes)} · {file.extension || file.inferred_format}</small>
          </button>
        ))}
        {!materialFiles.length && <EmptyState text={totalMaterialFiles ? t("noMatchingMaterials") : t("noData")} />}
      </div>
    </aside>
  );
}

function CorpusDatasetSummary({ profile, t }) {
  if (!profile) {
    return <EmptyState text={t("selectDatasetProfileToInspect")} />;
  }
  const readiness = trainingDatasetReadiness(profile, t);
  const ReadinessIcon = readiness.state === "ok" ? CheckCircle2 : readiness.state === "partial" ? Info : AlertTriangle;
  return (
    <section className="corpus-object-head">
      <span className="corpus-kicker">{t("currentTrainingDataset")}</span>
      <h2>{profile.name || profile.id}</h2>
      <div className={`corpus-readiness ${readiness.state}`}>
        <ReadinessIcon size={16} />
        <strong>{readiness.label}</strong>
        <span>{readiness.hint}</span>
      </div>
    </section>
  );
}

function DatasetSplitTable({ profile, selectedRole, setSelectedRole, t }) {
  if (!profile) return null;
  return (
    <section className="corpus-split-table">
      <header>
        <span className="corpus-kicker">{t("datasetComposition")}</span>
        <h2>{t("datasetSplitGroupTitle")}</h2>
      </header>
      <div className="corpus-split-table-head">
        <span>{t("datasetSplit")}</span>
        <span>{t("sampleRows")}</span>
        <span>{t("filePath")}</span>
        <span>{t("datasetStatus")}</span>
        <span>{t("action")}</span>
      </div>
      {["train", "val", "test"].map((role) => {
        const file = profileFile(profile, role);
        const ready = fileReady(file);
        return (
          <button key={role} className={selectedRole === role ? "corpus-split-row active" : "corpus-split-row"} onClick={() => ready && setSelectedRole(role)} disabled={!ready}>
            <strong>{roleLabel(role, t)}</strong>
            <span>{file?.rows ?? "-"}</span>
            <span title={file?.path || ""}>{file?.path || t("missing")}</span>
            <span className={ready ? "ok-text" : "warn-text"}>{ready ? t("ready") : t("missing")}</span>
            <span>{ready ? t("previewSamples") : role === "test" ? t("createMissingSplits") : "-"}</span>
          </button>
        );
      })}
    </section>
  );
}

function CorpusInspector({ profile, activePath, checkResult, runCheck, onEdit, onCopy, onCreate, onDelete, deletePending, t }) {
  const validationOk = profile?.validation?.ok;
  const openFolder = useMutation({ mutationFn: () => apiPost("/api/open-folder", { path: activePath }) });
  return (
    <aside className="corpus-inspector">
      <div className="corpus-inspector-list">
        <span>{t("taskType")}</span>
        <strong>{profile?.task_type || "-"}</strong>
        <span>{t("datasetFormat")}</span>
        <strong>{profile?.format || "-"}</strong>
        <span>{t("sampleRows")}</span>
        <strong>{profile?.total_rows ?? "-"} {profile?.total_rows != null ? t("rowsUnit") : ""}</strong>
        <span>{t("size")}</span>
        <strong>{formatBytes(profile?.total_size_bytes)}</strong>
        <span>{t("formatCheck")}</span>
        <strong className={validationOk ? "ok-text" : "warn-text"}>{validationOk ? t("validationPassed") : t("validationFailed")}</strong>
      </div>
      {checkResult && !checkResult.pending && (
        <div className="corpus-validation-toast" role="status" aria-live="polite">
          <ValidationBadge result={checkResult} t={t} />
        </div>
      )}
      <div className="corpus-inspector-actions">
        <button type="button" disabled={!activePath} onClick={runCheck}>{t("formatCheck")}</button>
        <button type="button" onClick={onEdit} disabled={!profile}>{t("editDatasetProfile")}</button>
        <button type="button" onClick={onCopy} disabled={!profile}>{t("copyDatasetProfile")}</button>
        <button type="button" onClick={onCreate}>{t("createDatasetProfile")}</button>
        <button type="button" disabled={!activePath || openFolder.isPending} onClick={() => openFolder.mutate()}>{t("openFolder")}</button>
        <button type="button" className="danger-link" disabled={!profile || deletePending} onClick={onDelete}>{t("delete")}</button>
      </div>
    </aside>
  );
}

function CorpusSampleReader({
  title,
  subtitle,
  activePath,
  preview,
  previewQuery,
  setPreviewQuery,
  previewLimit,
  setPreviewLimit,
  previewPage,
  setPreviewPage,
  t,
}) {
  return (
    <section className="corpus-wide-reader">
      <div className="corpus-reader-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <div className="corpus-reader-tools">
          <label className="compact-search">
            <Search size={15} />
            <input
              value={previewQuery}
              placeholder={t("searchSamples")}
              onChange={(event) => {
                setPreviewQuery(event.target.value);
                setPreviewPage(0);
              }}
            />
          </label>
          <SelectControl
            value={String(previewLimit)}
            onChange={(limit) => setPreviewLimit(Number(limit))}
            ariaLabel={t("samplesUnit")}
            options={[20, 50, 100].map((limit) => [String(limit), `${limit} ${t("samplesUnit")}`])}
          />
          <button className="secondary-button icon-only" title={t("previousPage")} disabled={!preview.data?.pagination?.has_previous} onClick={() => setPreviewPage((page) => Math.max(0, page - 1))}>
            <ChevronLeft size={16} />
          </button>
          <button className="secondary-button icon-only" title={t("nextPage")} disabled={!preview.data?.pagination?.has_next} onClick={() => setPreviewPage((page) => page + 1)}>
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
      <div className="reader-pagination">
        <span>{t("sampleRange", {
          start: preview.data?.pagination?.total ? previewPage * previewLimit + 1 : 0,
          end: previewPage * previewLimit + (preview.data?.samples?.length || 0),
          total: preview.data?.pagination?.total || 0,
        })}</span>
      </div>
      <div className="corpus-sample-list">
        {preview.isLoading && <EmptyState text={t("loadingCorpusSamples")} />}
        {!activePath && <EmptyState text={t("missingSplitHint")} />}
        {activePath && !preview.isLoading && !preview.data?.samples?.length && <EmptyState text={t("noData")} />}
        {activePath && preview.data?.samples?.map((sample) => (
          <CorpusSampleRow key={sample.line} sample={sample} t={t} />
        ))}
      </div>
    </section>
  );
}

function CorpusSampleRow({ sample, t }) {
  const normalized = sample.normalized || {};
  if (sample.error) {
    return <SampleCard sample={sample} t={t} />;
  }
  return (
    <article className="corpus-sample-row">
      <header>
        <span>{t("line", { line: sample.line })}</span>
        <strong>{normalized.task_type || "-"}</strong>
      </header>
      <div className="corpus-sample-instruction">
        <label>{t("instruction")}</label>
        <p>{normalized.instruction || "-"}</p>
      </div>
      <div className="corpus-sample-columns">
        <section>
          <label>{t("input")}</label>
          <p>{normalized.input || "-"}</p>
        </section>
        <section>
          <label>{t("output")}</label>
          <p>{normalized.output || "-"}</p>
        </section>
      </div>
    </article>
  );
}

function profileEditorInitialValue(editor) {
  const profile = editor?.profile || {};
  const copyMode = editor?.mode === "copy";
  return {
    id: copyMode ? `${profile.id || "dataset"}_copy` : profile.id || "",
    name: copyMode ? `${profile.name || profile.id || ""} copy` : profile.name || "",
    description: profile.description || "",
    task_type: profile.task_type || "rewrite",
    format: profile.format || "chatml_source",
    train_file: profile.train?.path || "",
    val_file: profile.val?.path || "",
    test_file: profile.test?.path || "",
  };
}

function ProfileEditorDrawer({ editor, files, saving, error, onClose, onSave, t }) {
  const [form, setForm] = useState(() => profileEditorInitialValue(editor));

  useEffect(() => {
    setForm(profileEditorInitialValue(editor));
  }, [editor]);

  if (!editor) return null;
  const fileOptions = [["", t("notSelected")], ...files.map((file) => [file.path, `${file.name} · ${file.rows ?? "-"} ${t("rowsUnit")}`])];
  const title = editor.mode === "edit"
    ? t("editDatasetProfile")
    : editor.mode === "copy"
      ? t("copyDatasetProfile")
      : t("createDatasetProfile");
  const canSave = Boolean(form.id.trim() && form.name.trim() && form.train_file);

  return (
    <ContextDrawer open title={title} subtitle={t("datasetProfileEditorHint")} onClose={onClose} t={t}>
      <div className="drawer-stack">
        <Panel title={t("profileIdentity")}>
          <div className="form-grid">
            <InputField label={t("profileId")} value={form.id} onChange={(id) => setForm({ ...form, id })} />
            <InputField label={t("name")} value={form.name} onChange={(name) => setForm({ ...form, name })} />
          </div>
          <InputField label={t("description")} value={form.description} onChange={(description) => setForm({ ...form, description })} />
          <div className="form-grid">
            <SelectField label={t("taskType")} value={form.task_type} onChange={(task_type) => setForm({ ...form, task_type })} options={[
              ["rewrite", t("rewriteType")],
              ["instruction", t("instructionType")],
              ["chat", t("chatType")],
              ["qa", "QA"],
              ["classification", "Classification"],
              ["dpo", t("dpoType")],
            ]} />
            <SelectField label={t("datasetFormat")} value={form.format} onChange={(format) => setForm({ ...form, format })} options={[
              ["chatml_source", "ChatML source"],
              ["localtune_v1", "LocalTune v1"],
              ["messages", "Messages"],
            ]} />
          </div>
        </Panel>
        <Panel title={t("datasetComposition")} subtitle={t("datasetProfileFileHint")}>
          <div className="form-grid">
            <SelectField label={t("trainData")} value={form.train_file} onChange={(train_file) => setForm({ ...form, train_file })} options={fileOptions} error={!form.train_file ? t("trainDataRequired") : ""} />
            <SelectField label={t("valData")} value={form.val_file} onChange={(val_file) => setForm({ ...form, val_file })} options={fileOptions} />
            <SelectField label={t("testData")} value={form.test_file} onChange={(test_file) => setForm({ ...form, test_file })} options={fileOptions} />
          </div>
        </Panel>
        {error && <div className="error-text">{apiErrorMessage(error, t)}</div>}
        <div className="drawer-footer-actions">
          <button className="secondary-button" onClick={onClose}>{t("cancel")}</button>
          <button className="primary-button" disabled={!canSave || saving} onClick={() => onSave(form)}>
            {saving ? <Loader2 className="spin" size={17} /> : <CheckCircle2 size={17} />}
            {t("save")}
          </button>
        </div>
      </div>
    </ContextDrawer>
  );
}

function DatasetSplitCard({ role, file, active, onSelect, t }) {
  const ready = fileReady(file);
  return (
    <article className={active ? "split-card active" : "split-card"}>
      <header>
        <div>
          <strong>{roleLabel(role, t)}</strong>
          <span>{file?.path || t("missing")}</span>
        </div>
        <span className={ready ? "status-pill completed" : "status-pill failed"}>
          {ready ? t("ready") : t("missing")}
        </span>
      </header>
      <div className="split-card-meta">
        <span>{ready ? `${file.rows ?? "-"} ${t("rowsUnit")}` : "-"}</span>
        <span>{formatBytes(file?.size_bytes)}</span>
        <span>{formatDate(file?.updated)}</span>
      </div>
      <div className="split-card-actions">
        <button className="secondary-button" disabled={!ready} onClick={onSelect}>
          <FileJson size={15} />
          {t("previewSamples")}
        </button>
        {ready && <OpenFolderButton path={file.path} label={t("openFolder")} />}
      </div>
    </article>
  );
}

function ProfileSplitPanel({ profile, missingRoles, onDone, t }) {
  const [valRatio, setValRatio] = useState("0.05");
  const [testRatio, setTestRatio] = useState("0.05");
  const [seed, setSeed] = useState("42");
  const split = useMutation({
    mutationFn: () => apiPost("/api/datasets/profile-split", {
      profile: profile.id,
      val_ratio: Number(valRatio),
      test_ratio: Number(testRatio),
      seed: Number(seed),
    }),
    onSuccess: () => onDone(),
  });
  const canSplit = fileReady(profile?.train);
  return (
    <Panel title={t("splitFromTrain")} subtitle={t("splitFromTrainHint")}>
      <div className="split-warning">
        <AlertTriangle size={16} />
        <span>{t("missingSplits", { roles: missingRoles.map((role) => roleLabel(role, t)).join(" / ") })}</span>
      </div>
      <div className="form-grid compact">
        <InputField label={t("valRatio")} value={valRatio} onChange={setValRatio} />
        <InputField label={t("testRatio")} value={testRatio} onChange={setTestRatio} />
        <InputField label={t("seed")} value={seed} onChange={setSeed} />
      </div>
      <button className="primary-button" disabled={!canSplit || split.isPending} onClick={() => split.mutate()}>
        {split.isPending ? <Loader2 className="spin" size={17} /> : <Scissors size={17} />}
        {t("createMissingSplits")}
      </button>
      {!canSplit && <div className="error-text">{t("trainDataRequired")}</div>}
      {split.data && <ValidationBadge result={split.data} t={t} />}
      {split.error && <div className="error-text">{apiErrorMessage(split.error, t)}</div>}
    </Panel>
  );
}

function MaterialContentPreview({ samples, t }) {
  const rawText = samples
    .map((sample) => sample.text || (sample.row ? JSON.stringify(sample.row, null, 2) : ""))
    .filter(Boolean)
    .join("\n\n---\n\n");
  const errors = samples.map((sample) => sample.error).filter(Boolean);
  return (
    <article className="sample-card material-card material-content-card">
      <header>
        <strong>{t("contentPreview")}</strong>
      </header>
      <div className="sample-section output material-content">
        <p>{rawText || "-"}</p>
      </div>
      {errors.map((error) => <div className="error-text" key={error}>{error}</div>)}
    </article>
  );
}
