import React, { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, Boxes, CheckCircle2, Globe2, Map } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { copy, navItems } from "./config/appConfig.jsx";
import { apiGet } from "./api.js";
import useHashRoute from "./hooks/useHashRoute.js";
import { ArtifactsPage } from "./pages/ArtifactsPage.jsx";
import { CorpusPage } from "./pages/CorpusPage.jsx";
import { EnvironmentPage } from "./pages/EnvironmentPage.jsx";
import { GoldenRunPage } from "./pages/GoldenRunPage.jsx";
import { HistoryPage } from "./pages/HistoryPage.jsx";
import { InferencePage } from "./pages/InferencePage.jsx";
import { ModelsPage } from "./pages/ModelsPage.jsx";
import { OverviewPage } from "./pages/OverviewPage.jsx";
import { TrainingPage } from "./pages/TrainingPage.jsx";

function initialLanguage() {
  const queryLang = new URLSearchParams(window.location.search).get("lang");
  if (queryLang === "en" || queryLang === "zh") return queryLang;
  try {
    const stored = window.localStorage?.getItem("localtune.lang");
    if (stored === "en" || stored === "zh") return stored;
  } catch {
    // Keep the default language when storage is unavailable.
  }
  return "zh";
}

function App() {
  const [lang, setLang] = useState(initialLanguage);
  const [route, setRoute] = useHashRoute();
  const quickStart = useQuery({
    queryKey: ["golden-path-status"],
    queryFn: () => apiGet("/api/golden-path/status"),
    refetchInterval: 6000,
  });
  const t = (key, values = {}) => {
    const template = copy[lang][key] || key;
    return Object.entries(values).reduce((text, [name, value]) => text.replaceAll(`{${name}}`, value), template);
  };
  const quickSteps = quickStart.data?.steps || [];
  const smokeDone = quickSteps.some((step) => step.id === "smoke" && step.status === "done");
  const pendingSteps = quickSteps.filter((step) => step.status !== "done").length;
  const nextStep = quickStart.data?.next_step;
  const hasExplicitRoute = window.location.hash.startsWith("#/");

  useEffect(() => {
    if (!hasExplicitRoute && route === "overview" && quickStart.data && !smokeDone) {
      setRoute("golden");
    }
  }, [hasExplicitRoute, quickStart.data, route, setRoute, smokeDone]);

  useEffect(() => {
    try {
      window.localStorage?.setItem("localtune.lang", lang);
    } catch {
      // Language selection is still usable without persistence.
    }
  }, [lang]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Boxes size={22} /></div>
          <div>
            <strong>{t("app")}</strong>
            <span>{t("sub")}</span>
          </div>
        </div>
        <button className={`sidebar-quickstart ${route === "golden" ? "active" : ""}`} onClick={() => setRoute("golden")}>
          <span className="sidebar-quickstart-icon"><Map size={18} /></span>
          <span className="sidebar-quickstart-copy">
            <strong>{t("goldenRunNav")}</strong>
            <small>{smokeDone ? t("quickStartDone") : nextStep ? t("quickStartNext", { step: t(`goldenStep_${nextStep.id}`) }) : t("quickStartSidebarHint")}</small>
          </span>
          <span className={smokeDone ? "sidebar-quickstart-status done" : "sidebar-quickstart-status pending"}>
            {smokeDone ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
            {smokeDone ? t("goldenStatus_done") : t("quickStartPending", { count: pendingSteps || "-" })}
          </span>
          <ArrowRight size={15} />
        </button>
        <nav className="nav-list">
          {navItems.map(([key, Icon]) => (
            <button key={key} className={route === key ? "active" : ""} onClick={() => setRoute(key)}>
              <Icon size={18} />
              <span>{t(key)}</span>
            </button>
          ))}
        </nav>
        <button className="language-toggle" onClick={() => setLang(lang === "zh" ? "en" : "zh")}>
          <Globe2 size={16} />
          {lang === "zh" ? "中文" : "English"}
        </button>
      </aside>
      <main className="workspace">
        {route === "overview" && <OverviewPage t={t} />}
        {route === "golden" && <GoldenRunPage t={t} lang={lang} />}
        {route === "models" && <ModelsPage t={t} lang={lang} />}
        {route === "corpus" && <CorpusPage t={t} />}
        {route === "training" && <TrainingPage t={t} />}
        {route === "artifacts" && <ArtifactsPage t={t} />}
        {route === "inference" && <InferencePage t={t} />}
        {route === "history" && <HistoryPage t={t} />}
        {route === "environment" && <EnvironmentPage t={t} />}
      </main>
    </div>
  );
}

export default App;
