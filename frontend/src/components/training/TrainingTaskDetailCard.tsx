import type { TrainingSummary, TrainingTask } from "../../types";
import { api } from "../../api/client";
import { formatDateTime, metricNumber, statusLabel } from "../../training/helpers";

interface Props {
  task: TrainingTask;
  summary?: TrainingSummary;
  logs?: string;
  busy: boolean;
  onClose: () => void;
  onStop: () => void;
  onResume: () => void;
}

export function TrainingTaskDetailCard({ task, summary, logs, busy, onClose, onStop, onResume }: Props) {
  const history = summary?.metrics.history ?? [];
  const lossSeries = [
    { key: "train_loss", label: "Loss", color: "#3157d5" },
    { key: "train_box_loss", label: "Box", color: "#3157d5" },
    { key: "train_cls_loss", label: "Cls", color: "#ef8d3c" },
    { key: "train_dfl_loss", label: "DFL", color: "#2a9d8f" },
    { key: "train_seg_loss", label: "Seg", color: "#9b5de5" },
  ].filter((series) => history.some((point) => typeof point[series.key] === "number"));
  const mapSeries = [{ key: "map50", label: "mAP50", color: "#3157d5" }];
  const value = (key: string) => {
    const fromSummary = metricNumber(summary?.metrics[key]);
    if (fromSummary !== undefined) return fromSummary.toFixed(3);
    const fromTask = metricNumber(task.metrics_json[key]);
    return fromTask !== undefined ? fromTask.toFixed(3) : "—";
  };

  const canStop = task.status === "running" || task.status === "pending";
  const canResume = !canStop && !!task.last_model_path;
  const currentLoss = metricNumber(summary?.metrics.loss) ?? metricNumber(task.metrics_json.loss);
  const timing = summary?.timing ?? {};

  return (
    <section className="training-ws-card training-detail-inline" id="training-task-detail">
      <div className="training-ws-card-head training-detail-head">
        <div>
          <div className="training-detail-title-row">
            <h2>{task.name}</h2>
            <span className={`training-ws-badge ${task.status}`}>{statusLabel(task.status)}</span>
          </div>
          <p className="training-ws-muted">
            {task.progress_epoch}/{task.progress_total_epochs || task.epochs} epochs · {task.progress_percent}%
            {" · "}创建于 {formatDateTime(task.created_at)}
          </p>
        </div>
        <div className="training-detail-actions">
          {canStop ? (
            <button className="button danger" disabled={busy} onClick={onStop}>停止训练</button>
          ) : (
            <>
              <button className="button primary" disabled={busy || !canResume} onClick={onResume}>
                {task.status === "completed" ? "从 last.pt 继续" : "恢复中断训练"}
              </button>
              {task.best_model_path && <a className="button" href={api.downloadCheckpointUrl(task.id, "best")}>best.pt</a>}
              {task.last_model_path && <a className="button" href={api.downloadCheckpointUrl(task.id, "last")}>last.pt</a>}
            </>
          )}
          <button className="button" onClick={onClose}>收起</button>
        </div>
      </div>

      {task.error_message && (
        <div className="validation invalid training-detail-error"><span>{task.error_message}</span></div>
      )}

      <div className="training-detail-metrics">
        <div className="metric-strip">
          <Metric label="mAP50" value={value("map50")} />
          <Metric label="mAP50-95" value={value("map50_95")} />
          <Metric label="Precision" value={value("precision")} />
          <Metric label="Recall" value={value("recall")} />
        </div>
        <div className="training-timing-grid">
          <Metric label="当前 Loss" value={formatNumber(currentLoss)} />
          <Metric label="批次耗时" value={formatSeconds(timing.batch_time_seconds)} />
          <Metric label="最近 Epoch" value={formatSeconds(timing.epoch_time_seconds)} />
          <Metric label="已耗时" value={formatSeconds(timing.elapsed_seconds)} />
          <Metric label="吞吐" value={timing.speed_it_per_sec ? `${timing.speed_it_per_sec.toFixed(2)} it/s` : "—"} />
        </div>
        {history.length > 1 && lossSeries.length > 0 && (
          <TrainingMetricChart title="Loss 曲线" count={history.length} history={history} series={lossSeries} />
        )}
        {history.length > 1 && history.some((point) => typeof point.map50 === "number") && (
          <TrainingMetricChart title="mAP50 趋势" count={history.length} history={history} series={mapSeries} max={1} />
        )}
        {summary?.export_stats?.counts ? (
          <p className="hint">数据导出：{formatExportStats(summary.export_stats)}</p>
        ) : null}
        {summary?.risks.length ? <p className="hint">风险提示：{summary.risks.join("、")}</p> : null}
      </div>

      <details className="config-snapshot">
        <summary>训练配置快照</summary>
        <pre>{JSON.stringify(summary?.training_config ?? {
          model: task.model_name,
          epochs: task.epochs,
          img_size: task.img_size,
          batch_size: task.batch_size,
          device: task.device,
          workers: task.workers,
          optimizer: task.optimizer,
          lr0: task.lr0,
          patience: task.patience,
        }, null, 2)}</pre>
      </details>

      <div className="log-box training-detail-log">{logs || "等待训练日志…"}</div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <span><small>{label}</small><strong>{value}</strong></span>;
}

function formatExportStats(stats: NonNullable<TrainingSummary["export_stats"]>): string {
  const exported = stats.counts ?? {};
  const annotated = stats.annotated_image_counts ?? {};
  const skipped = stats.skipped_image_counts ?? {};
  const splitText = (split: string) => `${split} 导出 ${exported[split] ?? 0}（已标注 ${annotated[split] ?? 0}）`;
  const parts = [splitText("train"), splitText("val")];
  const skippedTotal = Object.values(skipped).reduce((sum, value) => sum + value, 0);
  parts.push(`跳过未标注 ${skippedTotal}`);
  if (typeof stats.label_count === "number") parts.push(`标签 ${stats.label_count}`);
  return parts.join(" · ");
}

function TrainingMetricChart({
  title,
  count,
  history,
  series,
  max: fixedMax,
}: {
  title: string;
  count: number;
  history: Array<Record<string, number>>;
  series: Array<{ key: string; label: string; color: string }>;
  max?: number;
}) {
  const values = series.flatMap((item) => history.map((point) => point[item.key]).filter((value) => typeof value === "number"));
  const min = 0;
  const max = fixedMax ?? Math.max(...values, 1);
  return (
    <div className="metric-chart training-metric-chart">
      <div className="training-metric-chart-meta">
        <strong>{title}</strong>
        <small>{count} 个训练轮次</small>
        <div className="training-chart-legend">
          {series.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label}</span>)}
        </div>
      </div>
      <svg viewBox="0 0 100 80" preserveAspectRatio="none" role="img" aria-label={title}>
        <path d="M0 72H100" stroke="#dbe4ef" />
        {series.map((item) => (
          <path
            key={item.key}
            d={chartPath(history, item.key, min, max)}
            fill="none"
            stroke={item.color}
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
    </div>
  );
}

function chartPath(history: Array<Record<string, number>>, key: string, min: number, max: number): string {
  const span = Math.max(max - min, 0.00001);
  return history.map((point, index) => {
    const value = point[key];
    if (typeof value !== "number") return "";
    const x = index * (100 / Math.max(history.length - 1, 1));
    const y = 72 - ((value - min) / span) * 64;
    return `${index ? "L" : "M"}${x} ${Math.min(72, Math.max(8, y))}`;
  }).filter(Boolean).join(" ");
}

function formatNumber(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(3);
}

function formatSeconds(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (value < 1) return `${Math.round(value * 1000)}ms`;
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}
