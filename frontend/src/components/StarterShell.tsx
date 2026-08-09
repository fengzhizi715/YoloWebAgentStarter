import type { ReactNode } from "react";

export type StarterSection = "workspace" | "training" | "models";

interface Props {
  active: StarterSection;
  datasetName?: string;
  onNavigate: (section: StarterSection) => void;
  children: ReactNode;
}

const navigation: Array<{ id: StarterSection; label: string; icon: ReactNode }> = [
  { id: "workspace", label: "数据集", icon: <DatasetIcon /> },
  { id: "training", label: "训练", icon: <TrainingIcon /> },
  { id: "models", label: "模型", icon: <ModelIcon /> },
];

/**
 * 社区版工作台外壳。视觉和信息层级参考 YoloWebAgent，
 * 但刻意不包含 Enterprise 的认证、许可证、Agent 与权限入口。
 */
export function StarterShell({ active, datasetName, onNavigate, children }: Props) {
  return (
    <div className="starter-shell">
      <header className="starter-topbar">
        <button className="starter-topbar-brand" onClick={() => onNavigate("workspace")} aria-label="返回数据集">
          <span className="starter-topbar-mark">Y</span>
          <span>YoloWebAgent</span>
          <span className="starter-version">Starter</span>
        </button>
        <div className="starter-topbar-spacer" />
        {datasetName ? <span className="starter-current-dataset">当前数据集：{datasetName}</span> : null}
        <span className="starter-local-badge"><LocalIcon /> 本地模式</span>
      </header>
      <div className="starter-layout">
        <aside className="starter-sidebar">
          <div className="starter-workspace-brand">
            <span className="starter-workspace-mark">Y</span>
            <div><strong>我的工作区</strong><small>社区版</small></div>
          </div>
          <nav className="starter-menu" aria-label="主导航">
            {navigation.map((item) => (
              <button
                key={item.id}
                className={active === item.id ? "starter-menu-item active" : "starter-menu-item"}
                onClick={() => onNavigate(item.id)}
              >
                {item.icon}<span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="starter-sidebar-note">
            <strong>本地 YOLO 工作台</strong>
            <p>数据和模型仅保存在此设备。</p>
          </div>
          <div className="starter-sidebar-version">YoloWebAgent Starter · v0.1</div>
        </aside>
        <div className="starter-main">{children}</div>
      </div>
    </div>
  );
}

function DatasetIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-11Z" /><path d="m7.5 15 2.5-3 2 2.2 1.7-1.8 2.8 3.1" /><circle cx="9" cy="9" r="1.2" /></svg>; }
function TrainingIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 18V9m5 9V5m5 13v-7m4 7V8" /><path d="M3.5 20.5h17" /></svg>; }
function ModelIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 7 4v10l-7 4-7-4V7l7-4Z" /><path d="m5 7 7 4 7-4M12 11v10" /></svg>; }
function LocalIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4.5h14v11H5z" /><path d="M9 19.5h6M12 15.5v4" /></svg>; }
