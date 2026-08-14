import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { SamSettings, TrainingDevice } from "../types";
import type { AppLocale } from "../locale";
import { saveLocale } from "../locale";

type SettingsTab = "sam" | "language";

const copy = {
  zh: {
    eyebrow: "WORKSPACE SETTINGS", title: "设置", subtitle: "管理此设备上的辅助标注与工作台偏好。所有配置仅作用于本地 Starter。", local: "本地", samNav: "SAM 辅助", samNavDescription: "模型、设备与建议方式", languageNav: "语言", languageNavDescription: "工作台显示偏好", samTitle: "SAM 辅助标注", samSubtitle: "配置 segment 标注页使用的交互式 SAM 建议。建议结果仍需人工确认后保存。", enabled: "启用 SAM 辅助", model: "模型权重或模型 ID", modelHint: "可填写本地 .pt 路径或 Ultralytics 可识别的模型名，例如 sam_b.pt。", device: "推理设备", imageSize: "推理尺寸", fallback: "未配置模型时", box: "使用框形 review-only 建议", disabled: "禁用回退", save: "保存 SAM 设置", saved: "SAM 设置已保存", ready: "已配置真实模型", fallbackReady: "使用框形建议", disabledStatus: "已关闭", samFooter: "SAM 只对 segment 数据集开放。", languageTitle: "语言设置", languageSubtitle: "切换工作台界面语言。偏好只保存在本机浏览器中。", displayLanguage: "显示语言", chinese: "简体中文", english: "English", applied: "已立即应用", loading: "正在读取设置…", error: "设置读取失败", availableDevices: "当前设备", noDevice: "没有额外 GPU，仍可手动填写设备标识。", cuda: "CUDA", mps: "Apple GPU (MPS)", cpu: "CPU", auto: "自动选择",
  },
  en: {
    eyebrow: "WORKSPACE SETTINGS", title: "Settings", subtitle: "Manage assisted annotation and workspace preferences for this device. All settings apply to this local Starter only.", local: "Local", samNav: "SAM assist", samNavDescription: "Model, device, and proposal behavior", languageNav: "Language", languageNavDescription: "Workspace display preference", samTitle: "SAM assisted annotation", samSubtitle: "Configure interactive SAM proposals for segment annotation. Review every proposal before saving it.", enabled: "Enable SAM assist", model: "Model checkpoint or model id", modelHint: "Use a local .pt path or an Ultralytics model name such as sam_b.pt.", device: "Inference device", imageSize: "Inference size", fallback: "When no model is configured", box: "Use box-shaped review-only proposals", disabled: "Disable fallback", save: "Save SAM settings", saved: "SAM settings saved", ready: "Real model configured", fallbackReady: "Box proposals", disabledStatus: "Disabled", samFooter: "SAM assist is available for segment datasets only.", languageTitle: "Language settings", languageSubtitle: "Switch the workspace language. The preference is stored in this browser only.", displayLanguage: "Display language", chinese: "简体中文", english: "English", applied: "Applied immediately", loading: "Loading settings…", error: "Could not load settings", availableDevices: "Available devices", noDevice: "No extra GPU was detected; you can still enter a device id manually.", cuda: "CUDA", mps: "Apple GPU (MPS)", cpu: "CPU", auto: "Auto",
  },
} as const;

type SettingsText = (typeof copy)[AppLocale];

export function SettingsView({ tab, locale, onLocaleChange, onSamSettingsChange, onTabChange }: { tab: SettingsTab; locale: AppLocale; onLocaleChange: (locale: AppLocale) => void; onSamSettingsChange?: () => void; onTabChange: (tab: SettingsTab) => void }) {
  const text = copy[locale];
  return <main className="settings-layout">
    <header className="settings-page-head">
      <div><span className="eyebrow">{text.eyebrow}</span><h1>{text.title}</h1><p>{text.subtitle}</p></div>
      <span className="settings-page-badge">{text.local}</span>
    </header>
    <div className="settings-section-layout">
      <nav className="settings-section-nav" aria-label={text.title}>
        <SettingsNavigationItem active={tab === "sam"} icon={<SamIcon />} title={text.samNav} description={text.samNavDescription} badge={text.local} onClick={() => onTabChange("sam")} />
        <SettingsNavigationItem active={tab === "language"} icon={<LanguageIcon />} title={text.languageNav} description={text.languageNavDescription} onClick={() => onTabChange("language")} />
      </nav>
      <div className="settings-section-content">
        {tab === "sam" ? <SamSettingsPanel locale={locale} text={text} onSamSettingsChange={onSamSettingsChange} /> : <LanguageSettingsPanel locale={locale} text={text} onLocaleChange={onLocaleChange} />}
      </div>
    </div>
  </main>;
}

function SettingsNavigationItem({ active, icon, title, description, badge, onClick }: { active: boolean; icon: ReactNode; title: string; description: string; badge?: string; onClick: () => void }) {
  return <button type="button" className={`settings-section-nav-item ${active ? "active" : ""}`} aria-current={active ? "page" : undefined} onClick={onClick}>
    <span className="settings-nav-icon">{icon}</span><span className="settings-section-nav-copy"><strong>{title}</strong><small>{description}</small></span>{badge && <span className="settings-nav-badge">{badge}</span>}
  </button>;
}

function SamSettingsPanel({ locale, text, onSamSettingsChange }: { locale: AppLocale; text: SettingsText; onSamSettingsChange?: () => void }) {
  const [settings, setSettings] = useState<SamSettings>();
  const [devices, setDevices] = useState<TrainingDevice[]>([]);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([api.getSamSettings(), api.listTrainingDevices()])
      .then(([nextSettings, nextDevices]) => { setSettings(nextSettings); setDevices(nextDevices.items); })
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, [text.error]);

  const deviceOptions = useMemo(() => [
    { value: "auto", label: text.auto },
    { value: "cpu", label: text.cpu },
    ...devices.filter((device) => device.type === "mps").map(() => ({ value: "mps", label: text.mps })),
    ...(devices.some((device) => device.type === "cuda") ? [{ value: "cuda", label: text.cuda }] : []),
    ...devices.filter((device) => device.type === "cuda").map((device) => ({ value: String(device.index ?? 0), label: `${text.cuda}:${device.index ?? 0} · ${device.name}` })),
  ], [devices, text]);

  const save = async () => {
    if (!settings) return;
    setSaving(true); setError(""); setMessage("");
    try {
      setSettings(await api.updateSamSettings({ enabled: settings.enabled, model: settings.model, device: settings.device, img_size: settings.img_size, fallback_mode: settings.fallback_mode }));
      onSamSettingsChange?.();
      setMessage(text.saved);
    } catch (reason) { setError(reason instanceof Error ? reason.message : text.error); } finally { setSaving(false); }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;
  const status = settings.enabled ? (settings.model_configured ? text.ready : text.fallbackReady) : text.disabledStatus;
  return <section className="panel settings-card">
    <header className="settings-card-head"><div className="settings-card-icon"><SamIcon /></div><div><h2>{text.samTitle}</h2><p>{text.samSubtitle}</p></div><span className={`settings-status ${settings.enabled ? "ready" : "disabled"}`}>{status}</span></header>
    {error && <div className="validation invalid"><span>{error}</span></div>}
    {message && <div className="validation valid"><span>{message}</span></div>}
    <div className="settings-form-grid">
      <label className="settings-toggle"><input type="checkbox" checked={settings.enabled} onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })} /><span><strong>{text.enabled}</strong><small>{settings.model_configured ? settings.model : text.modelHint}</small></span></label>
      <label><span>{text.model}</span><input value={settings.model} placeholder="sam_b.pt" onChange={(event) => setSettings({ ...settings, model: event.target.value })} /><small>{text.modelHint}</small></label>
      <label><span>{text.device}</span><select value={settings.device} onChange={(event) => setSettings({ ...settings, device: event.target.value })}>{!deviceOptions.some((option) => option.value === settings.device) && <option value={settings.device}>{settings.device}</option>}{deviceOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
      <label><span>{text.imageSize}</span><input type="number" min={64} max={4096} step={64} value={settings.img_size} onChange={(event) => setSettings({ ...settings, img_size: Number(event.target.value) })} /></label>
      <label><span>{text.fallback}</span><select value={settings.fallback_mode} onChange={(event) => setSettings({ ...settings, fallback_mode: event.target.value as SamSettings["fallback_mode"] })}><option value="box">{text.box}</option><option value="disabled">{text.disabled}</option></select></label>
    </div>
    {!devices.some((device) => device.type !== "cpu") && <p className="settings-hint">{text.noDevice}</p>}
    <footer className="settings-footer"><span>{text.samFooter}</span><button className="button primary" disabled={saving} onClick={() => void save()}>{saving ? "…" : text.save}</button></footer>
  </section>;
}

function LanguageSettingsPanel({ locale, text, onLocaleChange }: { locale: AppLocale; text: SettingsText; onLocaleChange: (locale: AppLocale) => void }) {
  const change = (next: AppLocale) => { saveLocale(next); onLocaleChange(next); };
  return <section className="panel settings-card language-card">
    <header className="settings-card-head"><div className="settings-card-icon"><LanguageIcon /></div><div><h2>{text.languageTitle}</h2><p>{text.languageSubtitle}</p></div><span className="settings-status ready">{text.applied}</span></header>
    <div className="language-card-body"><strong>{text.displayLanguage}</strong><div className="language-segment" role="group" aria-label={text.displayLanguage}><button type="button" className={locale === "zh" ? "active" : ""} onClick={() => change("zh")}>{text.chinese}</button><button type="button" className={locale === "en" ? "active" : ""} onClick={() => change("en")}>{text.english}</button></div></div>
    <footer className="settings-footer"><span>{text.applied}</span></footer>
  </section>;
}

function SamIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5h16v13H4zM8 9l2.5 2.5L16 8m-8 7h8" /></svg>;
}

function LanguageIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8" /><path d="M4 12h16M12 4c2.2 2.1 3.3 4.8 3.3 8S14.2 17.9 12 20c-2.2-2.1-3.3-4.8-3.3-8S9.8 6.1 12 4" /></svg>;
}
