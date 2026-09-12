import type { ReactNode } from "react";
import type { AppLocale } from "../locale";
import { localeText } from "../locale";

export type StarterSection = "workspace" | "training" | "models" | "agent" | "logs" | "settings";

interface Props {
  active: StarterSection;
  datasetName?: string;
  onNavigate: (section: StarterSection) => void;
  locale?: AppLocale;
  children: ReactNode;
}

const navigation: Array<{ id: StarterSection; label: string; icon: ReactNode }> = [
  { id: "workspace", label: "数据集", icon: <DatasetIcon /> },
  { id: "training", label: "训练", icon: <TrainingIcon /> },
  { id: "models", label: "模型", icon: <ModelIcon /> },
  { id: "agent", label: "智能体", icon: <AgentIcon /> },
  { id: "logs", label: "日志", icon: <LogsIcon /> },
  { id: "settings", label: "设置", icon: <SettingsIcon /> },
];

/**
 * 社区版工作台外壳。视觉和信息层级参考 YoloWebAgent。
 * Agent MVP 为本地只读助手 + 人工确认提交；不含 Workflow / 无人值守入口。
 */
export function StarterShell({ active, datasetName, onNavigate, locale = "zh", children }: Props) {
  const text = localeText[locale];
  const labels: Record<StarterSection, string> = {
    workspace: text.datasets,
    training: text.training,
    models: text.models,
    agent: text.agent,
    settings: text.settingsGroup,
    logs: text.logs,
  };
  return (
    <div className="starter-shell">
      <header className="starter-topbar">
        <button className="starter-topbar-brand" onClick={() => onNavigate("workspace")} aria-label={text.backToDatasets}>
          <span className="starter-topbar-mark">Y</span>
          <span>YoloWebAgent</span>
          <span className="starter-version">Starter</span>
        </button>
        <div className="starter-topbar-spacer" />
        {datasetName ? <span className="starter-current-dataset">{text.currentDataset}{datasetName}</span> : null}
        <span className="starter-local-badge"><LocalIcon /> {text.localMode}</span>
      </header>
      <div className="starter-layout">
        <aside className="starter-sidebar">
          <div className="starter-workspace-brand">
            <span className="starter-workspace-mark">Y</span>
            <div><strong>{text.workspace}</strong><small>{text.community}</small></div>
          </div>
          <nav className="starter-menu" aria-label={text.mainNav}>
            <span className="starter-menu-group-label">{text.workspaceGroup}</span>
            {navigation.slice(0, 5).map((item) => <NavigationButton key={item.id} item={item} active={active} labels={labels} onNavigate={onNavigate} />)}
            <span className="starter-menu-group-label">{text.settingsGroup}</span>
            {navigation.slice(5).map((item) => <NavigationButton key={item.id} item={item} active={active} labels={labels} onNavigate={onNavigate} />)}
          </nav>
          <div className="starter-sidebar-note">
            <strong>{text.sidebarNoteTitle}</strong>
            <p>{text.sidebarNoteBody}</p>
          </div>
          <div className="starter-sidebar-version">YoloWebAgent Starter · v0.1</div>
        </aside>
        <div className="starter-main">{children}</div>
      </div>
    </div>
  );
}

function NavigationButton({ item, active, labels, onNavigate }: { item: (typeof navigation)[number]; active: StarterSection; labels: Record<StarterSection, string>; onNavigate: (section: StarterSection) => void }) {
  return <button className={active === item.id ? "starter-menu-item active" : "starter-menu-item"} onClick={() => onNavigate(item.id)}>
    {item.icon}<span>{labels[item.id]}</span>
  </button>;
}

function DatasetIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-11Z" /><path d="m7.5 15 2.5-3 2 2.2 1.7-1.8 2.8 3.1" /><circle cx="9" cy="9" r="1.2" /></svg>; }
function TrainingIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 18V9m5 9V5m5 13v-7m4 7V8" /><path d="M3.5 20.5h17" /></svg>; }
function ModelIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 7 4v10l-7 4-7-4V7l7-4Z" /><path d="m5 7 7 4 7-4M12 11v10" /></svg>; }
function LocalIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4.5h14v11H5z" /><path d="M9 19.5h6M12 15.5v4" /></svg>; }
function SettingsIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" /><path d="m19 13.5 1.2 1-.2 1.7-1.5 1-1.7-.6-1 1.1.1 1.8-1.5.8-1.4-1.1h-1.5L10 20.3l-1.6-.8.1-1.8-1-1.1-1.7.6-1.5-1 .2-1.7 1.2-1v-1.5l-1.2-1 .2-1.7 1.5-1 1.7.6 1-1.1-.1-1.8L10 5l1.4 1.1h1.5L14.3 5l1.6.8-.1 1.8 1 1.1 1.7-.6 1.5 1-.2 1.7-1.2 1Z" /></svg>; }
function LogsIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4.5h12v15H6z" /><path d="M9 8h6M9 12h6M9 16h4" /></svg>; }
function AgentIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z" /><path d="M5.5 19c.8-3.2 3.3-5 6.5-5s5.7 1.8 6.5 5" /><path d="M18 8.5h2.5M3.5 8.5H6M12 2v1.5" /></svg>; }
