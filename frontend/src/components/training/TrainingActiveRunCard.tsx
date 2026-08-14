import type { TrainingTask } from "../../types";
import { formatDateTime, formatShortId, metricNumber, statusLabel } from "../../training/helpers";
import type { ParsedTrainingLogs } from "../../training/trainingLogParse";
import { IconCpu, IconList, IconStop } from "./icons";

interface Props {
  task: TrainingTask;
  liveEpoch: number;
  liveProgressPct: number;
  liveLoss: number | undefined;
  liveEta: string;
  liveMap50: number | undefined;
  parsedLive: ParsedTrainingLogs;
  busy: boolean;
  onStop: () => void;
  onOpenDetail: () => void;
}

export function TrainingActiveRunCard({
  task,
  liveEpoch,
  liveProgressPct,
  liveLoss,
  liveEta,
  liveMap50,
  parsedLive,
  busy,
  onStop,
  onOpenDetail,
}: Props) {
  return (
    <section className="training-ws-card training-ws-control">
      <div className="training-ws-card-head">
        <div className="training-ws-control-title">
          <span className="training-ws-live-dot" aria-hidden />
          <div>
            <h2>活动训练</h2>
            <p className="training-ws-muted">实时进度来自任务状态与日志解析。</p>
          </div>
        </div>
        <p className="training-ws-config-line">
          {task.model_name} · {task.epochs} ep · batch {task.batch_size} · {task.device}
        </p>
      </div>

      <div className="training-ws-control-body">
        <div className="training-ws-control-main">
          <div className="training-ws-control-heading">
            <h3>{task.name}</h3>
            <span className={`training-ws-badge ${task.status}`}>{statusLabel(task.status)}</span>
          </div>
          <p className="training-ws-muted">
            开始于 {formatDateTime(task.started_at)} · ID {formatShortId(task.id)}
          </p>

          <div className="training-ws-progress-wrap">
            <div className="training-ws-progress-top">
              <span>Epoch {liveEpoch}/{task.progress_total_epochs || task.epochs}</span>
              <span>{liveProgressPct.toFixed(1)}%</span>
            </div>
            <div className="training-progress-track training-ws-progress-fat" role="progressbar" aria-valuenow={liveProgressPct}>
              <span style={{ width: `${Math.min(100, Math.max(0, liveProgressPct))}%` }} />
            </div>
          </div>

          <div className="training-ws-control-footer">
            <span>本机 · {task.device}</span>
            <span><IconCpu size={14} /> {parsedLive.speedItPerSec ?? "—"}</span>
          </div>

          <div className="training-ws-control-actions">
            <button type="button" className="training-ws-icon-btn" title="查看详情" onClick={onOpenDetail}>
              <IconList size={16} />
            </button>
            <button
              type="button"
              className="training-ws-icon-btn danger"
              disabled={busy || (task.status !== "running" && task.status !== "pending")}
              title="停止训练"
              onClick={onStop}
            >
              <IconStop size={15} />
            </button>
          </div>
        </div>

        <aside className="training-ws-control-metrics">
          <div>
            <span className="training-ws-metric-label">mAP50</span>
            <strong>{liveMap50 !== undefined ? liveMap50.toFixed(2) : "—"}</strong>
          </div>
          <div>
            <span className="training-ws-metric-label">当前 Loss</span>
            <strong>{liveLoss !== undefined ? liveLoss.toFixed(2) : "—"}</strong>
          </div>
          <div>
            <span className="training-ws-metric-label">剩余</span>
            <strong>{liveEta}</strong>
            <span className="training-ws-metric-hint">粗估</span>
          </div>
        </aside>
      </div>
    </section>
  );
}

export function liveMetricsFromTask(task: TrainingTask, parsed: ParsedTrainingLogs, summaryMap50?: number) {
  const lastRow = parsed.epochRows.length ? parsed.epochRows[parsed.epochRows.length - 1] : undefined;
  const liveEpoch = task.progress_epoch || lastRow?.epoch || 0;
  const liveProgressPct = task.progress_percent || (task.epochs ? (liveEpoch / task.epochs) * 100 : 0);
  const liveLoss = lastRow?.boxLoss;
  const liveMap50 = summaryMap50 ?? metricNumber(task.metrics_json?.map50);
  return { liveEpoch, liveProgressPct, liveLoss, liveMap50 };
}
