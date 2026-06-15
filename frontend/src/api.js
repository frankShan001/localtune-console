import { demoGet } from "./demoData.js";

const jsonHeaders = { "Content-Type": "application/json" };

export class ApiError extends Error {
  constructor(message, code = "REQUEST_FAILED", status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

function responseError(response, data) {
  return new ApiError(data.error || response.statusText, data.code || "REQUEST_FAILED", response.status);
}

export async function apiGet(path) {
  const demoData = demoGet(path);
  if (demoData) return demoData;
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) {
    throw responseError(response, data);
  }
  return data;
}

export async function apiPost(path, payload = {}) {
  return apiWrite(path, "POST", payload);
}

export async function apiPut(path, payload = {}) {
  return apiWrite(path, "PUT", payload);
}

export async function apiDelete(path, payload) {
  return apiWrite(path, "DELETE", payload);
}

async function apiWrite(path, method, payload) {
  const options = {
    method,
    headers: jsonHeaders,
  };
  if (payload !== undefined) {
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(path, {
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw responseError(response, data);
  }
  return data;
}

export function apiErrorMessage(error, t) {
  if (!error) return "";
  const key = `error_${error.code || "REQUEST_FAILED"}`;
  const translated = typeof t === "function" ? t(key) : t?.[key] || key;
  return translated && translated !== key ? translated : error.message || String(error);
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function shortPath(value) {
  if (!value) return "-";
  const text = String(value);
  return text.length > 64 ? `...${text.slice(-61)}` : text;
}
