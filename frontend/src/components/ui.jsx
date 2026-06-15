import React from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, ChevronDown, CircleHelp, FolderOpen, RefreshCw, X, XCircle } from "lucide-react";
import { apiPost } from "../api.js";

export function PageToolbar({
  icon: Icon,
  title,
  subtitle,
  stats = [],
  actions,
  onRefresh,
  refreshing = false,
  refreshLabel = "Refresh",
}) {
  return (
    <section className="page-toolbar">
      <div className="page-toolbar-copy">
        <strong className="page-toolbar-title">
          {Icon && <Icon size={18} />}
          {title}
        </strong>
        {subtitle && <span>{subtitle}</span>}
      </div>
      <div className="page-toolbar-controls">
        {stats.length > 0 && (
          <div className="page-summary-strip" aria-live="polite">
            {stats.map(({ icon: StatIcon, label, value, tone = "" }) => (
              <span className={tone} key={label}>
                {StatIcon && <StatIcon size={14} />}
                {label} <strong>{value}</strong>
              </span>
            ))}
          </div>
        )}
        {actions}
        {onRefresh && (
          <button className="ghost-button" disabled={refreshing} onClick={onRefresh}>
            <RefreshCw className={refreshing ? "spin" : ""} size={16} />
            {refreshLabel}
          </button>
        )}
      </div>
    </section>
  );
}

export function StatCard({ icon: Icon, label, value, detail, tone = "neutral" }) {
  return (
    <article className={`stat-card ${tone}`}>
      <Icon size={19} />
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

export function Panel({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={["panel", className].filter(Boolean).join(" ")}>
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {actions && <div className="panel-actions">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

export function ContextDrawer({ open, title, subtitle, actions, children, onClose, t = (key) => key }) {
  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="context-drawer-backdrop" onMouseDown={() => onClose?.()}>
      <aside className="context-drawer" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(event) => event.stopPropagation()}>
        <header className="context-drawer-head">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <div className="context-drawer-actions">
            {actions}
            <button className="secondary-button icon-only" title={t("close")} onClick={() => onClose?.()}>
              <X size={18} />
            </button>
          </div>
        </header>
        <div className="context-drawer-body">
          {children}
        </div>
      </aside>
    </div>
  );
}

export function TabBar({ value, onChange, items }) {
  return (
    <div className="tabbar">
      {items.map(([key, label, Icon, tone]) => (
        <button key={key} className={`${value === key ? "active" : ""} ${tone ? `tone-${tone}` : ""}`.trim()} onClick={() => onChange(key)}>
          <Icon size={16} /> {label}
        </button>
      ))}
    </div>
  );
}

export function InputField({ label, value, onChange, help, placeholder, error, hint, disabled = false }) {
  return (
    <label className={error ? "field error" : "field"}>
      <span>{label} {help && <Help text={help} />}</span>
      <input disabled={disabled} value={value ?? ""} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
      {(error || hint) && <small>{error || hint}</small>}
    </label>
  );
}

export function SelectControl({ value, onChange, options = [], ariaLabel, disabled = false, className = "" }) {
  const normalized = options.length ? options : [["", "-"]];
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef(null);
  const selected = normalized.find(([optionValue]) => optionValue === (value ?? "")) || normalized[0];
  const controlId = React.useId();

  React.useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className={["select-control-root", className].filter(Boolean).join(" ")} ref={rootRef}>
      <button
        type="button"
        className={open ? "select-control open" : "select-control"}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        aria-controls={open ? controlId : undefined}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <span>{selected?.[1] || "-"}</span>
        <ChevronDown size={15} />
      </button>
      {open && (
        <div className="select-menu" role="listbox" id={controlId} aria-label={ariaLabel}>
          {normalized.map(([optionValue, text], index) => (
            <button
              type="button"
              role="option"
              aria-selected={optionValue === (value ?? "")}
              className={optionValue === (value ?? "") ? "select-option selected" : "select-option"}
              key={`${optionValue}-${index}`}
              onClick={() => {
                onChange(optionValue);
                setOpen(false);
              }}
            >
              {text}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function SelectField({ label, value, onChange, options = [], help, error, hint }) {
  const labelId = React.useId();
  return (
    <div className={error ? "field select-field error" : "field select-field"}>
      <span id={labelId}>{label} {help && <Help text={help} />}</span>
      <SelectControl value={value} onChange={onChange} options={options} ariaLabel={label} />
      {(error || hint) && <small>{error || hint}</small>}
    </div>
  );
}

export function Help({ text }) {
  return (
    <span className="help" title={text}>
      <CircleHelp size={14} />
    </span>
  );
}

export function isPathLikeValue(value) {
  if (typeof value !== "string") return false;
  const normalized = value.trim();
  if (normalized.length < 28) return false;
  return /(^[A-Za-z]:[\\/])|(^\.{0,2}[\\/])|([\\/].+[\\/])/.test(normalized);
}

export function PropertyList({ rows }) {
  return (
    <dl className="property-list">
      {rows.map(([key, value]) => (
        <React.Fragment key={key}>
          <dt>{key}</dt>
          <dd className={isPathLikeValue(value) ? "path-value" : ""} title={value == null ? "" : String(value)}>
            {value == null || value === "" ? "-" : String(value)}
          </dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

export function SampleCard({ sample, t = (key, values = {}) => values.line ? `Line ${values.line}` : key }) {
  const normalized = sample.normalized || {};
  if (sample.error) {
    return (
      <article className="sample-card error">
        <header>{t("line", { line: sample.line })}</header>
        <p>{sample.error}</p>
        <pre>{sample.text}</pre>
      </article>
    );
  }
  return (
    <article className="sample-card">
      <header>
        <span>{t("line", { line: sample.line })}</span>
        <strong>{normalized.task_type || "-"}</strong>
      </header>
      <div className="sample-section">
        <label>{t("instruction")}</label>
        <p>{normalized.instruction || "-"}</p>
      </div>
      {normalized.input && <div className="sample-section">
        <label>{t("input")}</label>
        <p>{normalized.input}</p>
      </div>}
      <div className="sample-section output">
        <label>{t("output")}</label>
        <p>{normalized.output || "-"}</p>
      </div>
    </article>
  );
}

export function ValidationBadge({ result, t = (key) => ({ yes: "yes", no: "no", rowsUnit: "rows" }[key] || key) }) {
  const ok = result.ok !== false;
  return (
    <div className={ok ? "validation ok" : "validation bad"}>
      {ok ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
      <span>{ok ? t("validationPassed") : t("validationFailed")}</span>
      <small>{result.rows != null ? `${result.rows} ${t("rowsUnit")}` : result.error || result.output || ""}</small>
    </div>
  );
}

export function OpenFolderButton({ path, label }) {
  const open = useMutation({ mutationFn: () => apiPost("/api/open-folder", { path }) });
  return (
    <button className="secondary-button" disabled={!path || open.isPending} onClick={() => open.mutate()}>
      <FolderOpen size={16} /> {label}
    </button>
  );
}

export function ConfirmDialog({ open, title, message, confirmLabel, cancelLabel, pending, onCancel, onConfirm }) {
  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") onCancel?.();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;
  return (
    <div className="confirm-dialog-backdrop" onMouseDown={() => onCancel?.()}>
      <section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <AlertTriangle size={20} />
          <h2 id="confirm-dialog-title">{title}</h2>
        </header>
        <p>{message}</p>
        <footer>
          <button type="button" className="secondary-button" disabled={pending} onClick={onCancel}>{cancelLabel}</button>
          <button type="button" className="danger-button" disabled={pending} onClick={onConfirm}>{confirmLabel}</button>
        </footer>
      </section>
    </div>
  );
}

export function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}
