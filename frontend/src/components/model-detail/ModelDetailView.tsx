import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type {
  Dataset,
  ModelEvaluationRecord,
  ModelVersion,
  SplitName,
} from "../../types";
import { ModelEvaluationPanel } from "../ModelEvaluationPanel";
import {
  fileName,
  formatMetric,
  metricBarWidth,
  modelTaskLabel,
  newerFirst,
} from "../../models/helpers";

type DetailTab = "overview" | "evaluation" | "versions" | "metadata";

interface Props {
  dataset: Dataset;
  model: ModelVersion;
  models: ModelVersion[];
  evaluations: ModelEvaluationRecord[];
  busy: boolean;
  onBack: () => void;
  onSelectModel: (model: ModelVersion) => void;
  onTest: (model: ModelVersion) => void;
  onExportOnnx: () => void;
  onToggleArchive: () => void;
  onDelete: () => void;
  onEvaluate: (split: SplitName) => void;
  onSaved: (model: ModelVersion) => void;
}

export function ModelDetailView({
  dataset,
  model,
  models,
  evaluations,
  busy,
  onBack,
  onSelectModel,
  onTest,
  onExportOnnx,
  onToggleArchive,
  onDelete,
  onEvaluate,
  onSaved,
}: Props) {
  const [tab, setTab] = useState<DetailTab>("overview");
  const [name, setName] = useState(model.name);
  const [version, setVersion] = useState(model.version);
  const [notes, setNotes] = useState(model.notes);
  const siblings = useMemo(
    () => [...models].filter((item) => item.task_type === model.task_type).sort(newerFirst),
    [models, model.task_type],
  );

  useEffect(() => {
    setName(model.name);
    setVersion(model.version);
    setNotes(model.notes);
  }, [model.id, model.name, model.version, model.notes]);

  const save = async () => {
    const updated = await api.updateModel(model.id, { name: name.trim(), version: version.trim(), notes });
    onSaved(updated);
  };

  const lineage = [
    { id: "dataset", label: dataset.name },
    model.base_model ? { id: "base", label: model.base_model } : null,
    model.training_task_id ? { id: "train", label: "训练任务" } : null,
    { id: "artifact", label: `${model.artifact_type}.${model.format}`, highlight: true },
    { id: "file", label: fileName(model.model_path), highlight: true },
  ].filter(Boolean) as Array<{ id: string; label: string; highlight?: boolean }>;

  return (
    <div className="mdd-shell">
      <div className="mdd-topbar">
        <button className="button" onClick={onBack}>← 返回模型库</button>
        <div className="mdd-topbar-meta">
          <span className="models-chip">{dataset.name}</span>
          <span className="models-chip models-chip--accent">{modelTaskLabel(model)}</span>
        </div>
      </div>

      <header className="mdd-header">
        <div className="mdd-header-main">
          <span className="eyebrow">MODEL DETAIL / {model.task_type.toUpperCase()}</span>
          <h1>{model.name}</h1>
          <p className="muted">{model.format.toUpperCase()} · {model.artifact_type} · {model.source === "training_task" ? "受管训练产物" : "导出产物"}</p>
          <div className="mdd-header-tags">
            <span className={`badge-premium ${model.status === "active" ? "badge-active" : "badge-archived"}`}>
              {model.status === "active" ? "启用" : "已归档"}
            </span>
            <span className="models-chip">v{model.version}</span>
            {model.base_model && <span className="models-chip">{model.base_model}</span>}
          </div>
        </div>
        <div className="mdd-header-actions">
          <a className="button" href={api.downloadModelUrl(model.id)}>下载 {model.format.toUpperCase()}</a>
          {model.format === "pt" && (
            <>
              <button className="button primary" disabled={busy} onClick={() => onTest(model)}>快速测试</button>
              <button className="button" disabled={busy} onClick={onExportOnnx}>导出 ONNX</button>
            </>
          )}
          <button className="button" disabled={busy} onClick={onToggleArchive}>
            {model.status === "active" ? "归档" : "恢复"}
          </button>
          <button className="button danger" disabled={busy} onClick={onDelete}>删除</button>
        </div>
      </header>

      <div className="mdd-summary-grid">
        <div className="mdd-summary-card">
          <small>状态</small>
          <strong>{model.status === "active" ? "可用" : "已归档"}</strong>
          <span>{model.format.toUpperCase()} · {model.artifact_type}</span>
        </div>
        <div className="mdd-summary-card">
          <small>mAP50</small>
          <strong>{formatMetric(model.map50)}</strong>
          <span>mAP50-95 {formatMetric(model.map50_95)}</span>
        </div>
        <div className="mdd-summary-card">
          <small>导出格式</small>
          <strong>{model.format === "onnx" ? "ONNX FP32" : "PyTorch PT"}</strong>
          <span>{model.engine_type}</span>
        </div>
        <div className="mdd-summary-card">
          <small>来源</small>
          <strong>{model.source === "training_task" ? "受管训练" : "导出产物"}</strong>
          <span>{model.created_at.slice(0, 16).replace("T", " ")}</span>
        </div>
      </div>

      <div className="mdd-layout">
        <div className="mdd-main">
          <div className="mdd-tabs" role="tablist">
            {([
              ["overview", "概览"],
              ["evaluation", "评估"],
              ["versions", "版本历史"],
              ["metadata", "元数据"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                className={`mdd-tab${tab === id ? " active" : ""}`}
                aria-selected={tab === id}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "overview" && (
            <div className="mdd-overview" style={{ display: "grid", gap: 16 }}>
              <section className="mdd-panel">
                <header className="mdd-panel-head"><h3>模型血缘</h3></header>
                <div className="mdd-panel-body">
                  <div className="mdd-lineage">
                    {lineage.map((node, index) => (
                      <div key={node.id} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                        {index > 0 && <span className="mdd-lineage-arrow">›</span>}
                        <span className={`mdd-lineage-node${node.highlight ? " highlight" : ""}`}>{node.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              <div className="mdd-metrics-row">
                <article className="mdd-panel">
                  <div className="mdd-panel-body">
                    <p className="mdd-metric-section-label">Precision / Recall</p>
                    <div className="mdd-pr-grid">
                      <div>
                        <strong>{formatMetric(model.precision)}</strong>
                        <div className="mdd-metric-bar">
                          <span className="blue" style={{ width: metricBarWidth(model.precision) }} />
                        </div>
                        <span>Precision</span>
                      </div>
                      <div>
                        <strong>{formatMetric(model.recall)}</strong>
                        <div className="mdd-metric-bar">
                          <span className="green" style={{ width: metricBarWidth(model.recall) }} />
                        </div>
                        <span>Recall</span>
                      </div>
                    </div>
                  </div>
                </article>
                <article className="mdd-panel">
                  <div className="mdd-panel-body">
                    <p className="mdd-metric-section-label">Detection quality</p>
                    <div className="mdd-map-block">
                      <strong className="mdd-map-value">{formatMetric(model.map50)}</strong>
                      <p className="mdd-muted">mAP50-95 · {formatMetric(model.map50_95)}</p>
                    </div>
                  </div>
                </article>
              </div>

              <section className="mdd-panel">
                <header className="mdd-panel-head"><h3>下一步</h3></header>
                <div className="mdd-panel-body">
                  <div className="mdd-next-actions">
                    {model.format === "pt" && (
                      <>
                        <button className="button primary" disabled={busy} onClick={() => onTest(model)}>运行快速测试</button>
                        <button className="button" disabled={busy} onClick={() => setTab("evaluation")}>创建 split 评估</button>
                        <button className="button" disabled={busy} onClick={onExportOnnx}>导出 ONNX FP32</button>
                      </>
                    )}
                    {model.format === "onnx" && (
                      <a className="button primary" href={api.downloadModelUrl(model.id)}>下载 ONNX</a>
                    )}
                  </div>
                </div>
              </section>
            </div>
          )}

          {tab === "evaluation" && (
            model.format === "pt"
              ? <ModelEvaluationPanel model={model} evaluations={evaluations} busy={busy} onEvaluate={onEvaluate} />
              : <section className="mdd-panel"><div className="mdd-panel-body"><p className="mdd-muted">ONNX 产物请回到源 PT 模型创建评估任务。</p></div></section>
          )}

          {tab === "versions" && (
            <section className="mdd-panel">
              <header className="mdd-panel-head"><h3>同数据集版本</h3></header>
              <div className="mdd-panel-body" style={{ padding: 0 }}>
                <table className="mdd-version-table">
                  <thead>
                    <tr>
                      <th>名称</th>
                      <th>格式</th>
                      <th>mAP50</th>
                      <th>状态</th>
                      <th>创建</th>
                    </tr>
                  </thead>
                  <tbody>
                    {siblings.map((item) => (
                      <tr key={item.id} className={item.id === model.id ? "active" : undefined}>
                        <td>
                          <button type="button" className="linkish" onClick={() => onSelectModel(item)}>
                            {item.name}
                          </button>
                        </td>
                        <td>{item.format.toUpperCase()} · {item.artifact_type}</td>
                        <td>{formatMetric(item.map50)}</td>
                        <td>{item.status === "active" ? "启用" : "归档"}</td>
                        <td>{item.created_at.slice(0, 16).replace("T", " ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {tab === "metadata" && (
            <section className="mdd-panel">
              <header className="mdd-panel-head"><h3>编辑元数据</h3></header>
              <div className="mdd-panel-body">
                <div className="mdd-form">
                  <label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
                  <label>版本<input value={version} onChange={(event) => setVersion(event.target.value)} /></label>
                  <label>备注<textarea value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
                  <button className="button primary" disabled={busy || !name.trim() || !version.trim()} onClick={() => void save()}>
                    保存元数据
                  </button>
                </div>
              </div>
            </section>
          )}
        </div>

        <aside className="mdd-sidebar">
          <div className="mdd-sidebar-card">
            <h3>就绪检查</h3>
            <ul className="mdd-ready-list">
              <li><span>权重文件</span><span className="ok">已登记</span></li>
              <li><span>训练来源</span><span className={model.training_task_id ? "ok" : "pending"}>{model.training_task_id ? "已关联" : "无"}</span></li>
              <li><span>评估指标</span><span className={typeof model.map50 === "number" ? "ok" : "pending"}>{typeof model.map50 === "number" ? "有" : "待评估"}</span></li>
              <li><span>ONNX</span><span className={model.format === "onnx" || models.some((item) => item.source_model_id === model.id && item.format === "onnx") ? "ok" : "pending"}>{model.format === "onnx" ? "当前即为 ONNX" : models.some((item) => item.source_model_id === model.id && item.format === "onnx") ? "已导出" : "未导出"}</span></li>
            </ul>
          </div>
          <div className="mdd-sidebar-card">
            <h3>文件路径</h3>
            <p className="mdd-path">{model.model_path}</p>
          </div>
          <div className="mdd-sidebar-card">
            <h3>备注预览</h3>
            <p className="mdd-muted">{model.notes.trim() || "暂无备注"}</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
