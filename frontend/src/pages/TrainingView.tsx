import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { api } from "../api/client";
import { TrainingActiveRunCard, liveMetricsFromTask } from "../components/training/TrainingActiveRunCard";
import { TrainingComputeRail } from "../components/training/TrainingComputeRail";
import { TrainingConfigPanel } from "../components/training/TrainingConfigPanel";
import { TrainingHistoryGrid } from "../components/training/TrainingHistoryGrid";
import { TrainingTaskDetailCard } from "../components/training/TrainingTaskDetailCard";
import {
  PRESETS,
  applyTaskToForm,
  defaultForm,
  defaultModelFor,
  estimateMinutes,
  pickPreset,
  resolveDeviceString,
  taskTypeLabel,
  type HistoryFilter,
  type PresetId,
  type TrainingFormState,
} from "../training/helpers";
import { formatEtaSeconds, parseTrainingLogs } from "../training/trainingLogParse";
import type { Dataset, TaskType, TrainingDevice, TrainingLog, TrainingSummary, TrainingTask } from "../types";

interface Props {
  datasets: Dataset[];
  dataset: Dataset;
  onDatasetChange: (dataset: Dataset) => void;
  onOpenModels?: () => void;
}

const DRAFT_KEY = "ywa.training.draft";

export function TrainingView({ datasets, dataset, onDatasetChange, onOpenModels }: Props) {
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [detailTaskId, setDetailTaskId] = useState<string>();
  const [logs, setLogs] = useState<TrainingLog>();
  const [summary, setSummary] = useState<TrainingSummary>();
  const [form, setForm] = useState<TrainingFormState>(() => defaultForm(dataset.task_type));
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [devices, setDevices] = useState<TrainingDevice[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [splitRecovery, setSplitRecovery] = useState(false);
  const [expSearch, setExpSearch] = useState("");
  const [expFilter, setExpFilter] = useState<HistoryFilter>("all");

  const preset = pickPreset(form);
  const activeTask = useMemo(
    () => tasks.find((task) => task.status === "running" || task.status === "pending"),
    [tasks],
  );
  const historyTasks = useMemo(
    () => tasks.filter((task) => task.id !== activeTask?.id),
    [tasks, activeTask],
  );
  const detailTask = useMemo(
    () => tasks.find((task) => task.id === detailTaskId) ?? (detailTaskId ? activeTask : undefined),
    [tasks, detailTaskId, activeTask],
  );

  const filteredHistory = useMemo(() => {
    const q = expSearch.trim().toLowerCase();
    return historyTasks
      .filter((task) => {
        if (expFilter === "completed" && task.status !== "completed") return false;
        if (expFilter === "failed" && task.status !== "failed") return false;
        if (expFilter === "stopped" && task.status !== "stopped") return false;
        if (!q) return true;
        return task.name.toLowerCase().includes(q)
          || task.model_name.toLowerCase().includes(q)
          || task.device.toLowerCase().includes(q);
      })
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  }, [historyTasks, expSearch, expFilter]);

  const refresh = useCallback(async () => {
    const result = await api.listTrainingTasks(dataset.id);
    setTasks(result.items);
  }, [dataset.id]);

  useEffect(() => {
    refresh().catch((reason) => setError(errorMessage(reason)));
  }, [refresh]);

  useEffect(() => {
    if (typeof api.listTrainingDevices !== "function") return;
    api.listTrainingDevices().then((result) => {
      setDevices(result.items);
      setForm((prev) => {
        if (prev.gpu_ids.length) return prev;
        const first = result.items.find((item) => item.type === "cuda" && item.index !== null);
        return first ? { ...prev, gpu_ids: [String(first.index)] } : prev;
      });
    }).catch((reason) => setError(errorMessage(reason)));
  }, []);

  useEffect(() => {
    setForm(defaultForm(dataset.task_type));
    setDetailTaskId(undefined);
    setLogs(undefined);
    setSummary(undefined);
    setError("");
    setNotice("");
    setSplitRecovery(false);
  }, [dataset.id, dataset.task_type]);

  useEffect(() => {
    if (!activeTask) return;
    const timer = window.setInterval(() => refresh().catch((reason) => setError(errorMessage(reason))), 1500);
    return () => window.clearInterval(timer);
  }, [refresh, activeTask?.id, activeTask?.status]);

  useEffect(() => {
    const task = detailTask ?? activeTask;
    if (!task) {
      setLogs(undefined);
      setSummary(undefined);
      return;
    }
    const taskId = task.id;
    const active = task.status === "pending" || task.status === "running";
    let cancelled = false;
    const loadDetail = async () => {
      try {
        const [nextLogs, nextSummary] = await Promise.all([
          api.getTrainingLogs(taskId),
          api.getTrainingSummary(taskId),
        ]);
        if (!cancelled) {
          setLogs(nextLogs);
          setSummary(nextSummary);
        }
      } catch (reason) {
        if (!cancelled) setError(errorMessage(reason));
      }
    };
    void loadDetail();
    if (!active) return () => { cancelled = true; };
    const timer = window.setInterval(() => void loadDetail(), 2500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [detailTask?.id, detailTask?.status, activeTask?.id, activeTask?.status]);

  const applyPreset = (id: PresetId) => {
    const next = PRESETS[id];
    setForm((prev) => ({
      ...prev,
      epochs: next.epochs,
      batch_size: next.batch_size,
      img_size: next.img_size,
      patience: String(next.patience),
      model: prev.model || defaultModelFor(dataset.task_type),
    }));
  };

  const canSubmit = !(form.device_type === "cuda" && (!form.gpu_ids.length || (form.compute_mode === "multi" && form.gpu_ids.length < 2)));

  const riskText = dataset.image_count < 2
    ? "图片不足 2 张时无法同时构成 train / val，创建任务可能失败。"
    : `将使用已持久化的 split 训练 ${taskTypeLabel(dataset.task_type)} 模型；请确认 train 与 val 均非空。`;

  const summaryBottomBar = `${form.model} · ${form.epochs} ep · batch ${form.batch_size} · ${form.img_size}px · ${resolveDeviceString(form)} · 约 ${estimateMinutes(form.epochs, dataset.image_count, form.batch_size, form.device_type === "cuda" ? Math.max(form.gpu_ids.length, 1) : 1)} 分钟`;

  const start = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    setSplitRecovery(false);
    try {
      const patience = form.patience.trim() ? Number(form.patience) : undefined;
      const lr0 = form.lr0.trim() ? Number(form.lr0) : undefined;
      const task = await api.createTrainingTask({
        dataset_id: dataset.id,
        name: form.name.trim() || "local-training",
        model: form.model.trim() || defaultModelFor(dataset.task_type),
        task_type: dataset.task_type as TaskType,
        epochs: form.epochs,
        img_size: form.img_size,
        batch_size: form.batch_size,
        device: resolveDeviceString(form),
        workers: form.workers,
        seed: form.seed,
        val_ratio: form.val_ratio,
        optimizer: form.optimizer.trim() || undefined,
        lr0: Number.isFinite(lr0) ? lr0 : undefined,
        patience: Number.isFinite(patience) ? patience : undefined,
      });
      setTasks((items) => [task, ...items]);
      setDetailTaskId(task.id);
      setNotice("训练任务已创建并进入队列。");
    } catch (reason) {
      const message = errorMessage(reason);
      setError(message);
      setSplitRecovery(message.includes("at least one train image and one val image"));
    } finally {
      setBusy(false);
    }
  };

  const autoSplitForTraining = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await api.autoSplitImages(dataset.id, { train_ratio: 0.8, val_ratio: 0.2, test_ratio: 0, seed: 42 });
      setSplitRecovery(false);
      setError(`已按 80/20 分配 ${result.updated} 张图片：train ${result.split_counts.train}，val ${result.split_counts.val}。现在可以重新创建训练。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const stop = async (taskId: string) => {
    setBusy(true);
    try {
      await api.stopTrainingTask(taskId);
      await refresh();
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const resume = async (task: TrainingTask) => {
    setBusy(true);
    setError("");
    try {
      const restoreEpochState = task.status !== "completed";
      const next = await api.resumeTrainingTask(task.id, restoreEpochState ? { resume_epoch: true } : { epochs: form.epochs, resume_epoch: false });
      setTasks((items) => [next, ...items]);
      setDetailTaskId(next.id);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const saveDraft = () => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify({ datasetId: dataset.id, form }));
      setNotice("草稿配置已保存到本机浏览器。");
    } catch {
      setError("无法写入本地草稿。");
    }
  };

  const parsedLive = useMemo(
    () => parseTrainingLogs(logs?.task_id === activeTask?.id ? (logs?.logs ?? "") : "", activeTask?.epochs ?? 0),
    [logs, activeTask?.id, activeTask?.epochs],
  );
  const activeLive = activeTask
    ? liveMetricsFromTask(
      activeTask,
      parsedLive,
      summary?.task_id === activeTask.id ? (typeof summary.metrics.map50 === "number" ? summary.metrics.map50 : undefined) : undefined,
    )
    : undefined;
  const liveEta = (() => {
    if (!activeTask || !activeLive) return "—";
    const remaining = Math.max(activeTask.epochs - activeLive.liveEpoch, 0);
    if (!remaining) return "即将完成";
    const perEpochSec = 45;
    return formatEtaSeconds(remaining * perEpochSec);
  })();

  return (
    <main className="training-layout training-workspace-page">
      <header className="training-page-header">
        <div className="training-page-header-main">
          <div>
            <span className="eyebrow">LOCAL TRAINING / {dataset.task_type.toUpperCase()}</span>
            <h1>训练工作区</h1>
            <p className="muted">复用已保存的 train / val / test split，在本机运行 Ultralytics YOLO。</p>
          </div>
        </div>
        <div className="training-page-header-meta" aria-label="当前数据集摘要">
          <span className="training-chip"><strong>{dataset.name}</strong></span>
          <span className="training-chip">{dataset.image_count} 张图片</span>
          <span className="training-chip">{dataset.class_count} 类</span>
          <span className="training-chip training-chip--accent">{taskTypeLabel(dataset.task_type)}</span>
        </div>
      </header>

      {notice && <p className="training-workspace-notice">{notice}</p>}
      {error && (
        <div className={`validation ${splitRecovery ? "invalid" : "valid"}`}>
          <strong>{splitRecovery ? "训练尚未创建" : "提示"}</strong>
          <span>{error}</span>
          {splitRecovery && (
            <div className="training-split-recovery">
              <small>这会重新分配该数据集全部图片的持久化 split；请确认当前 split 不需要保留。</small>
              <button className="button" disabled={busy || dataset.image_count < 2} onClick={() => void autoSplitForTraining()}>
                自动分配 train / val
              </button>
              {dataset.image_count < 2 && <small>至少需要两张图片，才能同时创建 train 与 val split。</small>}
            </div>
          )}
        </div>
      )}

      <div className="training-dataset-grid">
        <div className="training-dataset-main training-workspace">
          <TrainingConfigPanel
            datasets={datasets}
            dataset={dataset}
            form={form}
            preset={preset}
            advancedOpen={advancedOpen}
            summaryBottomBar={summaryBottomBar}
            onDatasetChange={onDatasetChange}
            onFormChange={setForm}
            onPreset={applyPreset}
            onAdvancedOpenChange={setAdvancedOpen}
            onSubmit={(event) => void start(event)}
          />

          {activeTask && activeLive && (
            <TrainingActiveRunCard
              task={activeTask}
              liveEpoch={activeLive.liveEpoch}
              liveProgressPct={activeLive.liveProgressPct}
              liveLoss={activeLive.liveLoss}
              liveEta={liveEta}
              liveMap50={activeLive.liveMap50}
              parsedLive={parsedLive}
              busy={busy}
              onStop={() => void stop(activeTask.id)}
              onOpenDetail={() => {
                setDetailTaskId(activeTask.id);
                window.requestAnimationFrame(() => document.getElementById("training-task-detail")?.scrollIntoView({ behavior: "smooth" }));
              }}
            />
          )}

          <TrainingHistoryGrid
            historyTasks={historyTasks}
            filteredHistory={filteredHistory}
            selectedTaskId={detailTaskId}
            expSearch={expSearch}
            expFilter={expFilter}
            onSearchChange={setExpSearch}
            onFilterChange={setExpFilter}
            onApplyTask={(task) => {
              setForm(applyTaskToForm(task));
              setAdvancedOpen(true);
              setNotice(`已填入「${task.name}」的配置，可直接调整后提交。`);
              document.getElementById("training-config-anchor")?.scrollIntoView({ behavior: "smooth" });
            }}
            onOpenDetail={(task) => {
              setDetailTaskId(task.id);
              window.requestAnimationFrame(() => document.getElementById("training-task-detail")?.scrollIntoView({ behavior: "smooth" }));
            }}
            onOpenModels={onOpenModels}
            onScrollToConfig={() => document.getElementById("training-config-anchor")?.scrollIntoView({ behavior: "smooth" })}
          />

          {detailTask && (
            <TrainingTaskDetailCard
              task={detailTask}
              summary={summary?.task_id === detailTask.id ? summary : undefined}
              logs={logs?.task_id === detailTask.id ? logs.logs : undefined}
              busy={busy}
              onClose={() => setDetailTaskId(undefined)}
              onStop={() => void stop(detailTask.id)}
              onResume={() => void resume(detailTask)}
            />
          )}
        </div>

        <TrainingComputeRail
          dataset={dataset}
          form={form}
          devices={devices}
          busy={busy}
          canSubmit={canSubmit}
          riskText={riskText}
          onFormChange={setForm}
          onSaveDraft={saveDraft}
        />
      </div>
    </main>
  );
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。";
}
