import type { ModelVersion } from "../../types";
import { artifactTagsFor, formatMetric } from "../../models/helpers";
import { formatRelativeTime } from "../../training/helpers";
import { IconPlay } from "../training/icons";

interface Props {
  model: ModelVersion;
  models: ModelVersion[];
  datasetName: string;
  selected: boolean;
  busy: boolean;
  onToggleSelect: (id: string) => void;
  onTest: (model: ModelVersion) => void;
  onOpenDetail: (model: ModelVersion) => void;
  onExportOnnx: (model: ModelVersion) => void;
  onDelete: (model: ModelVersion) => void;
}

export function ModelLibraryCard({
  model,
  models,
  datasetName,
  selected,
  busy,
  onToggleSelect,
  onTest,
  onOpenDetail,
  onExportOnnx,
  onDelete,
}: Props) {
  const tags = artifactTagsFor(model, models);
  const hasMap = typeof model.map50 === "number";
  const when = formatRelativeTime(model.created_at);
  const tone = model.status === "archived"
    ? "archived"
    : model.format === "onnx"
      ? "onnx"
      : model.artifact_type === "best"
        ? "best"
        : "default";

  return (
    <article className={`premium-card premium-card--${tone}${selected ? " active" : ""}`}>
      <div className="card-header-premium">
        <div className="model-card-badges">
          <span className={`badge-premium badge-${tone === "best" ? "active" : tone === "onnx" ? "onnx" : tone === "archived" ? "archived" : "checkpoint"}`}>
            {model.artifact_type.toUpperCase()}
          </span>
          <span className={`model-card-status ${model.status}`}>
            {model.status === "active" ? "启用" : "已归档"}
          </span>
        </div>
        <label className="model-card-check">
          <input
            type="checkbox"
            className="card-checkbox"
            checked={selected}
            onChange={() => onToggleSelect(model.id)}
            aria-label={`选择 ${model.name}`}
          />
        </label>
      </div>

      <div className="card-title-group">
        <button type="button" className="model-card-title-btn" onClick={() => onOpenDetail(model)}>
          <h3 title={model.name}>{model.name}</h3>
        </button>
        <div className="model-card-dataset" title={datasetName}>
          <span className="model-card-dataset-label">数据集</span>
          <span className="model-card-dataset-name">{datasetName || "未关联数据集"}</span>
        </div>
        <p className="model-card-time" title={model.created_at}>{when}</p>

        <div className="model-card-chips">
          <span>v{model.version}</span>
          {model.base_model && <span title={model.base_model}>{model.base_model}</span>}
        </div>

        <div className="card-metric-main">
          <div>
            <span className="label">mAP50</span>
            <p className="model-card-metric-hint">{hasMap ? "来自训练/评估摘要" : "暂无评估指标"}</p>
          </div>
          <span className="value">{formatMetric(model.map50)}</span>
        </div>
        <div className="card-progress-bar">
          <div className="card-progress-fill" style={{ width: `${hasMap ? (model.map50 || 0) * 100 : 0}%` }} />
        </div>
      </div>

      <div className="card-mini-stats">
        <div className="mini-stat">
          <div className="label">Precision</div>
          <div className="value">{formatMetric(model.precision)}</div>
        </div>
        <div className="mini-stat">
          <div className="label">Recall</div>
          <div className="value">{formatMetric(model.recall)}</div>
        </div>
        <div className="mini-stat">
          <div className="label">mAP50-95</div>
          <div className="value">{formatMetric(model.map50_95)}</div>
        </div>
      </div>

      <div className="model-artifact-tags" aria-label="产物标签">
        {tags.map((tag) => (
          <span key={tag.label} className={`model-artifact-tag model-artifact-tag--${tag.tone}`}>
            {tag.label}
          </span>
        ))}
      </div>

      <footer className="model-card-footer">
        {model.format === "pt" ? (
          <>
            <button type="button" className="model-action-btn model-action-btn--primary" disabled={busy} onClick={() => onTest(model)}>
              <IconPlay size={14} /> 测试
            </button>
            <button type="button" className="model-action-btn model-action-btn--ghost" disabled={busy} onClick={() => onExportOnnx(model)}>
              导出 ONNX
            </button>
          </>
        ) : (
          <button type="button" className="model-action-btn model-action-btn--primary" onClick={() => onOpenDetail(model)}>
            查看详情
          </button>
        )}
        <div className="model-card-util-actions">
          {model.format === "pt" && (
            <button type="button" className="model-action-btn model-action-btn--ghost" onClick={() => onOpenDetail(model)}>
              详情
            </button>
          )}
          <button type="button" className="model-action-btn model-action-btn--danger" disabled={busy} onClick={() => onDelete(model)}>
            删除
          </button>
        </div>
      </footer>
    </article>
  );
}
