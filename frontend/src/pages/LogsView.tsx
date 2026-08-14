import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { RuntimeLogResponse } from "../types";
import type { AppLocale } from "../locale";

const text = {
  zh: { eyebrow: "OPERATIONS", title: "日志", subtitle: "查看本地后端运行日志，定位训练、标注和服务问题。", refresh: "刷新", search: "搜索日志内容…", all: "全部", lines: "行数", viewer: "运行日志", path: "日志文件", empty: "暂无日志。", loaded: "已加载", copy: "复制路径", copied: "路径已复制", total: "总行数", info: "信息", warn: "警告", error: "错误" },
  en: { eyebrow: "OPERATIONS", title: "Logs", subtitle: "Review local backend runtime logs for training, annotation, and service issues.", refresh: "Refresh", search: "Search log content…", all: "All", lines: "Lines", viewer: "Runtime logs", path: "Log file", empty: "No logs yet.", loaded: "Loaded", copy: "Copy path", copied: "Path copied", total: "Total", info: "Info", warn: "Warnings", error: "Errors" },
} as const;

export function LogsView({ locale }: { locale: AppLocale }) {
  const copy = text[locale];
  const [result, setResult] = useState<RuntimeLogResponse>({ path: "", level: null, lines: [] });
  const [level, setLevel] = useState("");
  const [lineCount, setLineCount] = useState(300);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true); setError("");
    try { setResult(await api.runtimeLogs({ lines: lineCount, level: level || undefined })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Request failed"); }
    finally { setLoading(false); }
  };
  useEffect(() => { void load(); }, [level, lineCount]);
  const visibleLines = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query ? result.lines.filter((line) => line.toLowerCase().includes(query)) : result.lines;
  }, [result.lines, search]);
  const counts = useMemo(() => ({ info: result.lines.filter((line) => line.includes(" INFO ")).length, warn: result.lines.filter((line) => line.includes(" WARNING ")).length, error: result.lines.filter((line) => line.includes(" ERROR ")).length }), [result.lines]);
  const copyPath = async () => { if (result.path) await navigator.clipboard?.writeText(result.path); };

  return <main className="logs-layout">
    <header className="logs-head"><div><span className="eyebrow">{copy.eyebrow}</span><h1>{copy.title}</h1><p>{copy.subtitle}</p></div><div className="header-actions"><button className="button" onClick={() => void load()} disabled={loading}>{copy.refresh}</button><button className="button" onClick={() => void copyPath()} disabled={!result.path}>{copy.copy}</button></div></header>
    {error && <div className="validation invalid"><span>{error}</span></div>}
    <section className="logs-summary"><span><small>{copy.total}</small><strong>{result.lines.length}</strong></span><span><small>{copy.info}</small><strong>{counts.info}</strong></span><span><small>{copy.warn}</small><strong>{counts.warn}</strong></span><span><small>{copy.error}</small><strong>{counts.error}</strong></span></section>
    <section className="panel logs-toolbar"><label>{copy.search}<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={copy.search} /></label><label>{copy.lines}<select value={lineCount} onChange={(event) => setLineCount(Number(event.target.value))}><option value={100}>100</option><option value={300}>300</option><option value={1000}>1000</option></select></label><div className="log-levels"><button className={!level ? "active" : ""} onClick={() => setLevel("")}>{copy.all}</button>{["INFO", "WARNING", "ERROR"].map((item) => <button key={item} className={level === item ? "active" : ""} onClick={() => setLevel(item)}>{item}</button>)}</div></section>
    <section className="panel logs-panel"><header><div><h2>{copy.viewer}</h2><p>{copy.loaded} {visibleLines.length}{search ? ` / ${result.lines.length}` : ""}</p></div><code title={result.path}>{copy.path}: {result.path || "—"}</code></header>{loading ? <p className="muted">…</p> : visibleLines.length ? <pre>{visibleLines.join("\n")}</pre> : <p className="logs-empty">{copy.empty}</p>}</section>
  </main>;
}
