import type { TrainingTask } from "../../types";
import {
  formatDuration,
  formatRelativeTime,
  metricNumber,
  statusLabel,
  type HistoryFilter,
} from "../../training/helpers";
import { IconPlus, IconSearch } from "./icons";

interface Props {
  historyTasks: TrainingTask[];
  filteredHistory: TrainingTask[];
  selectedTaskId?: string;
  expSearch: string;
  expFilter: HistoryFilter;
  onSearchChange: (value: string) => void;
  onFilterChange: (value: HistoryFilter) => void;
  onApplyTask: (task: TrainingTask) => void;
  onOpenDetail: (task: TrainingTask) => void;
  onOpenModels: (() => void) | undefined;
  onScrollToConfig: () => void;
}

export function TrainingHistoryGrid({
  historyTasks,
  filteredHistory,
  selectedTaskId,
  expSearch,
  expFilter,
  onSearchChange,
  onFilterChange,
  onApplyTask,
  onOpenDetail,
  onOpenModels,
  onScrollToConfig,
}: Props) {
  const searchQuery = expSearch.trim();
  const counts = {
    all: historyTasks.length,
    completed: historyTasks.filter((task) => task.status === "completed").length,
    failed: historyTasks.filter((task) => task.status === "failed").length,
    stopped: historyTasks.filter((task) => task.status === "stopped").length,
  };
  const filters: Array<{ id: HistoryFilter; label: string; count: number }> = [
    { id: "all", label: "全部", count: counts.all },
    { id: "completed", label: "已完成", count: counts.completed },
    { id: "failed", label: "失败", count: counts.failed },
    { id: "stopped", label: "已停止", count: counts.stopped },
  ];

  const emptyFilteredMessage = (() => {
    if (searchQuery && expFilter !== "all") return "没有同时匹配筛选与搜索的实验。";
    if (searchQuery) return `没有名称包含「${searchQuery}」的实验。`;
    if (expFilter !== "all") return "当前筛选下没有实验。";
    return null;
  })();

  const showEmpty = !historyTasks.length;
  const showFilteredEmpty = historyTasks.length > 0 && !filteredHistory.length;

  return (
    <section id="experiment-history" className="training-ws-card training-ws-history">
      <div className="training-ws-history-head">
        <div className="training-ws-history-head-row">
          <div>
            <h2>实验历史</h2>
            <p className="training-ws-muted">查看过往训练结果，一键回填配置或打开日志。</p>
          </div>
          <span className="training-ws-history-count">{counts.all} 条记录</span>
        </div>

        <div className="training-ws-history-toolbar">
          <div className="training-ws-history-filters" role="tablist" aria-label="实验筛选">
            {filters.map((filter) => (
              <button
                key={filter.id}
                type="button"
                role="tab"
                aria-selected={expFilter === filter.id}
                className={`training-ws-history-filter${expFilter === filter.id ? " active" : ""}`}
                onClick={() => onFilterChange(filter.id)}
              >
                {filter.label}
                <em>{filter.count}</em>
              </button>
            ))}
          </div>
          <label className="training-ws-history-search">
            <IconSearch size={15} />
            <input
              type="search"
              value={expSearch}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="搜索名称或模型…"
              autoComplete="off"
            />
          </label>
        </div>
      </div>

      {showEmpty ? (
        <div className="training-ws-history-empty">
          <div className="training-ws-history-empty-icon" aria-hidden>+</div>
          <strong>还没有实验记录</strong>
          <p>配置参数后点击侧栏「开始训练」，完成后会出现在这里。</p>
          <button type="button" className="button primary" onClick={onScrollToConfig}>去配置训练</button>
        </div>
      ) : showFilteredEmpty ? (
        <div className="training-ws-history-empty training-ws-history-empty--filtered">
          <strong>没有匹配的实验</strong>
          <p>{emptyFilteredMessage}</p>
          <button type="button" className="button" onClick={() => { onFilterChange("all"); onSearchChange(""); }}>
            清除筛选
          </button>
        </div>
      ) : (
        <div className="training-ws-history-grid">
          {filteredHistory.map((task) => (
            <ExperimentCard
              key={task.id}
              task={task}
              selected={selectedTaskId === task.id}
              onApplyTask={onApplyTask}
              onOpenDetail={onOpenDetail}
              onOpenModels={onOpenModels}
            />
          ))}

          <button type="button" className="training-ws-new-exp" onClick={onScrollToConfig}>
            <IconPlus />
            <span>新建实验</span>
            <span className="training-ws-muted-inline">回到上方配置并提交</span>
          </button>
        </div>
      )}
    </section>
  );
}

function ExperimentCard({
  task,
  selected,
  onApplyTask,
  onOpenDetail,
  onOpenModels,
}: {
  task: TrainingTask;
  selected: boolean;
  onApplyTask: (task: TrainingTask) => void;
  onOpenDetail: (task: TrainingTask) => void;
  onOpenModels: (() => void) | undefined;
}) {
  const map50 = metricNumber(task.metrics_json?.map50);
  const precision = metricNumber(task.metrics_json?.precision);
  const recall = metricNumber(task.metrics_json?.recall);
  const hasMetrics = map50 !== undefined || precision !== undefined || recall !== undefined;
  const duration = formatDuration(task.started_at, task.finished_at);
  const when = formatRelativeTime(task.finished_at || task.started_at || task.created_at);
  const progress = Math.min(100, Math.max(0, task.progress_percent || 0));
  const showProgress = task.status === "stopped" || task.status === "failed" || (task.status === "completed" && progress > 0);

  return (
    <article className={`training-ws-exp-card ${task.status}${selected ? " selected" : ""}`}>
      <header>
        <div className="training-ws-exp-title">
          <h3 title={task.name}>{task.name}</h3>
          <span className={`training-ws-badge ${task.status}`}>{statusLabel(task.status)}</span>
        </div>
        <p className="training-ws-exp-time" title={task.finished_at || task.created_at}>{when}</p>
      </header>

      <div className="training-ws-exp-chips">
        <span>{task.model_name}</span>
        <span>{task.epochs} ep</span>
        <span>batch {task.batch_size}</span>
        <span>{task.img_size}px</span>
        <span>{task.device}</span>
        {duration && <span>{duration}</span>}
      </div>

      {showProgress && (
        <div className="training-ws-exp-progress">
          <div className="training-ws-exp-progress-top">
            <span>进度 {task.progress_epoch}/{task.progress_total_epochs || task.epochs}</span>
            <span>{progress.toFixed(0)}%</span>
          </div>
          <div className="training-progress-track">
            <span style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {task.status === "failed" && task.error_message && (
        <div className="training-ws-exp-error" title={task.error_message}>{task.error_message}</div>
      )}

      {hasMetrics && (
        <div className="training-ws-exp-metrics">
          <div>
            <span>mAP50</span>
            <strong>{fmt(map50)}</strong>
          </div>
          <div>
            <span>Precision</span>
            <strong>{fmt(precision)}</strong>
          </div>
          <div>
            <span>Recall</span>
            <strong>{fmt(recall)}</strong>
          </div>
        </div>
      )}

      {!hasMetrics && task.status === "completed" && (
        <p className="training-ws-exp-summary">训练已完成，指标尚未写入摘要。</p>
      )}

      <footer className="training-ws-exp-actions">
        <button type="button" className="training-ws-btn-ghost" onClick={() => onOpenDetail(task)}>
          查看详情
        </button>
        <button type="button" className="training-ws-btn-ghost" onClick={() => onApplyTask(task)}>
          重试配置
        </button>
        {task.status === "completed" && onOpenModels && (
          <button type="button" className="training-ws-btn-solid" onClick={onOpenModels}>
            模型库
          </button>
        )}
      </footer>
    </article>
  );
}

function fmt(value: number | undefined): string {
  return value !== undefined ? value.toFixed(2) : "—";
}
