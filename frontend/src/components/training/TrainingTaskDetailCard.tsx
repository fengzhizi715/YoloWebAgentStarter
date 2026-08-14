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
  const path = history.map((point, index) => {
    const x = index * (100 / Math.max(history.length - 1, 1));
    const y = 72 - (point.map50 ?? 0) * 64;
    return `${index ? "L" : "M"}${x} ${y}`;
  }).join(" ");
  const value = (key: string) => {
    const fromSummary = metricNumber(summary?.metrics[key]);
    if (fromSummary !== undefined) return fromSummary.toFixed(3);
    const fromTask = metricNumber(task.metrics_json[key]);
    return fromTask !== undefined ? fromTask.toFixed(3) : "—";
  };

  const canStop = task.status === "running" || task.status === "pending";
  const canResume = !canStop && !!task.last_model_path;

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
        {history.length > 1 && (
          <div className="metric-chart">
            <div>
              <strong>mAP50 趋势</strong>
              <small>{history.length} 个训练轮次</small>
            </div>
            <svg viewBox="0 0 100 80" preserveAspectRatio="none">
              <path d="M0 72H100" stroke="#dbe4ef" />
              <path d={path} fill="none" stroke="#3157d5" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            </svg>
          </div>
        )}
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
