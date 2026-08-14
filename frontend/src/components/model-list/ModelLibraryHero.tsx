import type { Dataset, ModelVersion } from "../../types";
import { formatMetric, modelTaskLabel } from "../../models/helpers";
import { formatRelativeTime } from "../../training/helpers";
import { IconPlay } from "../training/icons";

interface Props {
  bestModel: ModelVersion;
  dataset: Dataset;
  busy: boolean;
  onTest: (model: ModelVersion) => void;
  onOpenDetail: (model: ModelVersion) => void;
  onExportOnnx: (model: ModelVersion) => void;
}

export function ModelLibraryHero({ bestModel, dataset, busy, onTest, onOpenDetail, onExportOnnx }: Props) {
  const when = formatRelativeTime(bestModel.created_at);

  return (
    <section className="hero-section">
      <div className="section-header">
        <div className="title-group">
          <h2>当前最佳模型</h2>
          <p>按本数据集已登记的 mAP50 选出，便于快速推理、评估与导出。</p>
        </div>
        <span className="badge-premium badge-active">BEST · MAP50</span>
      </div>

      <div className="best-model-card">
        <div className="hero-image">
          <div className="hero-overlay">
            <span className="label">推荐推理权重</span>
            <h3 title={bestModel.name}>{bestModel.name}</h3>
            <div className="hero-overlay-meta">
              <span>{bestModel.format.toUpperCase()}</span>
              <span>{bestModel.artifact_type.toUpperCase()}</span>
              <span>v{bestModel.version}</span>
            </div>
          </div>
        </div>

        <div className="hero-stats">
          <div className="stats-grid-premium">
            <div className="main-stat">
              <div className="label">mAP50</div>
              <div className="stat-value-group">
                <span className="value">{formatMetric(bestModel.map50)}</span>
                <span className="hero-secondary-metric">mAP50-95 {formatMetric(bestModel.map50_95)}</span>
              </div>
              <p className="hero-recommend-copy">本数据集当前首选推理权重</p>
              <div className="card-progress-bar hero-progress">
                <div className="card-progress-fill" style={{ width: `${(bestModel.map50 || 0) * 100}%` }} />
              </div>
            </div>

            <div className="hero-side-stats">
              <div className="mini-stat">
                <div className="label">Precision</div>
                <div className="value">{formatMetric(bestModel.precision)}</div>
                <div className="usage-bar">
                  <div className="usage-fill" style={{ width: `${(bestModel.precision || 0) * 100}%` }} />
                </div>
              </div>
              <div className="mini-stat">
                <div className="label">Recall</div>
                <div className="value">{formatMetric(bestModel.recall)}</div>
                <div className="usage-bar">
                  <div className="usage-fill usage-fill--violet" style={{ width: `${(bestModel.recall || 0) * 100}%` }} />
                </div>
              </div>
            </div>
          </div>

          <div className="metadata-row">
            <div className="meta-item">
              <div className="meta-icon" aria-hidden>D</div>
              <div className="meta-info">
                <span>数据集</span>
                <strong>{dataset.name}</strong>
              </div>
            </div>
            <div className="meta-item">
              <div className="meta-icon" aria-hidden>T</div>
              <div className="meta-info">
                <span>任务</span>
                <strong>{modelTaskLabel(bestModel)}</strong>
              </div>
            </div>
            <div className="meta-item">
              <div className="meta-icon" aria-hidden>⏱</div>
              <div className="meta-info">
                <span>创建</span>
                <strong title={bestModel.created_at}>{when}</strong>
              </div>
            </div>
          </div>

          <div className="hero-actions-premium">
            <div className="hero-actions-main">
              {bestModel.format === "pt" && (
                <button type="button" className="btn-primary-premium" disabled={busy} onClick={() => onTest(bestModel)}>
                  <IconPlay size={16} /> 快速测试
                </button>
              )}
              {bestModel.format === "pt" && (
                <button type="button" className="btn-accent-premium" disabled={busy} onClick={() => onExportOnnx(bestModel)}>
                  导出 ONNX
                </button>
              )}
              <button type="button" className="btn-ghost-premium" onClick={() => onOpenDetail(bestModel)}>
                查看详情
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
