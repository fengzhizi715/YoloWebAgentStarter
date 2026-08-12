import type { SampleGroup } from "./evaluationDetailHelpers";
import { formatEvalMetric } from "./evaluationDetailHelpers";

// Community port of upstream EvaluationDetailPanels.tsx; pose and router-specific actions are excluded.
export function EvaluationDetailMetrics({ metrics }: { metrics: Record<string, number> }) {
  return <section className="evaluation-metrics">{Object.entries(metrics).length ? Object.entries(metrics).map(([key, value]) => <MetricCard key={key} label={key} value={value} />) : <p>暂无指标。</p>}</section>;
}

export function EvaluationDetailSummary({ summary }: { summary: { missed_detection: number; false_positive: number; low_confidence: number; imageCount: number } }) {
  return <section className="evaluation-summary"><SummaryCard label="missed_detection" value={summary.missed_detection} /><SummaryCard label="false_positive" value={summary.false_positive} /><SummaryCard label="low_confidence" value={summary.low_confidence} /><SummaryCard label="images" value={summary.imageCount} /></section>;
}

type EvaluationArtifactName = "pr_curve" | "box_pr_curve" | "mask_pr_curve" | "confusion_matrix";

export function EvaluationDetailArtifacts({ taskId, prCurvePath, boxPrCurvePath, maskPrCurvePath, confusionMatrixPath, artifactUrl }: { taskId: string; prCurvePath?: string | null; boxPrCurvePath?: string | null; maskPrCurvePath?: string | null; confusionMatrixPath?: string | null; artifactUrl: (taskId: string, name: EvaluationArtifactName) => string }) {
  const hasSplitCurves = Boolean(boxPrCurvePath || maskPrCurvePath);
  return <section className="evaluation-artifacts">{hasSplitCurves ? <><ArtifactCard title="Box PR Curve" url={boxPrCurvePath ? artifactUrl(taskId, "box_pr_curve") : ""} /><ArtifactCard title="Mask PR Curve" url={maskPrCurvePath ? artifactUrl(taskId, "mask_pr_curve") : ""} /></> : <ArtifactCard title="PR Curve" url={prCurvePath ? artifactUrl(taskId, "pr_curve") : ""} />}<ArtifactCard title="Confusion Matrix" url={confusionMatrixPath ? artifactUrl(taskId, "confusion_matrix") : ""} /></section>;
}

export function EvaluationDetailSamples({ groups }: { groups: SampleGroup[] }) {
  return <section className="evaluation-samples"><h4>错误样本（{groups.reduce((total, group) => total + group.items.length, 0)}）</h4>{groups.map((group) => <article key={group.imageKey}><div><span className={`evaluation-sample-badge is-${group.primaryType}`}>{group.primaryType}</span><strong>{group.imageFile}</strong></div><p>{group.preview}</p><small>{group.items.length} 条 · classes {group.classIndexes.join(", ") || "-"}{group.bestConfidence != null ? ` · max conf ${group.bestConfidence.toFixed(3)}` : ""}</small></article>)}{!groups.length && <p>暂无错误样本。</p>}</section>;
}

export function EvaluationDetailLogs({ logs }: { logs: string }) {
  return <details className="evaluation-logs"><summary>评估日志</summary><pre>{logs || "暂无日志。"}</pre></details>;
}

function MetricCard({ label, value }: { label: string; value?: number }) {
  return <article><span>{label}</span><strong>{formatEvalMetric(value)}</strong></article>;
}

function SummaryCard({ label, value }: { label: string; value: number }) {
  return <article><span>{label}</span><strong>{value}</strong></article>;
}

function ArtifactCard({ title, url }: { title: string; url: string }) {
  return <figure><figcaption>{title}</figcaption>{url ? <img src={url} alt={title} /> : <p>产物将在评估完成后显示。</p>}</figure>;
}
