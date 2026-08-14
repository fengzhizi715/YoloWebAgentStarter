import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Dataset, TaskType, TrainingDevice, TrainingLog, TrainingSummary, TrainingTask } from "../types";

interface Props {
  datasets: Dataset[];
  dataset: Dataset;
  onDatasetChange: (dataset: Dataset) => void;
  onBack: () => void;
}

export function TrainingView({ datasets, dataset, onDatasetChange, onBack }: Props) {
  const defaultModel = defaultModelFor(dataset.task_type);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<TrainingTask>();
  const [logs, setLogs] = useState<TrainingLog>();
  const [summary, setSummary] = useState<TrainingSummary>();
  const [name, setName] = useState("local-training");
  const [model, setModel] = useState(defaultModel);
  const [epochs, setEpochs] = useState(50);
  const [imgSize, setImgSize] = useState(640);
  const [batchSize, setBatchSize] = useState(16);
  const [device, setDevice] = useState("auto");
  const [devices, setDevices] = useState<TrainingDevice[]>([]);
  const [gpuMode, setGpuMode] = useState<"single" | "multi">("single");
  const [gpuIds, setGpuIds] = useState<string[]>([]);
  const [workers, setWorkers] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [splitRecovery, setSplitRecovery] = useState(false);

  const refresh = useCallback(async () => {
    const result = await api.listTrainingTasks(dataset.id);
    setTasks(result.items);
    setSelectedTask((current) => result.items.find((item) => item.id === current?.id) ?? result.items[0]);
  }, [dataset.id]);

  useEffect(() => {
    refresh().catch((reason) => setError(errorMessage(reason)));
  }, [refresh]);

  useEffect(() => {
    if (typeof api.listTrainingDevices !== "function") return;
    api.listTrainingDevices().then((result) => {
      setDevices(result.items);
      const first = result.items.find((item) => item.type === "cuda" && item.index !== null);
      if (first && !gpuIds.length) setGpuIds([String(first.index)]);
    }).catch((reason) => setError(errorMessage(reason)));
  }, []);

  useEffect(() => {
    setModel(defaultModelFor(dataset.task_type));
    setSelectedTask(undefined);
    setLogs(undefined);
    setSummary(undefined);
  }, [dataset.id, dataset.task_type]);

  useEffect(() => {
    const active = selectedTask && (selectedTask.status === "pending" || selectedTask.status === "running");
    if (!active) return;
    const timer = window.setInterval(() => refresh().catch((reason) => setError(errorMessage(reason))), 1500);
    return () => window.clearInterval(timer);
  }, [refresh, selectedTask]);

  useEffect(() => {
    if (!selectedTask) { setLogs(undefined); setSummary(undefined); return; }
    const taskId = selectedTask.id;
    const active = selectedTask.status === "pending" || selectedTask.status === "running";
    let cancelled = false;
    const loadDetail = async () => {
      try {
        const [nextLogs, nextSummary] = await Promise.all([api.getTrainingLogs(taskId), api.getTrainingSummary(taskId)]);
        if (!cancelled) { setLogs(nextLogs); setSummary(nextSummary); }
      } catch (reason) {
        if (!cancelled) setError(errorMessage(reason));
      }
    };
    void loadDetail();
    if (!active) return () => { cancelled = true; };
    const timer = window.setInterval(() => void loadDetail(), 2500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [selectedTask?.id, selectedTask?.status]);

  const start = async () => {
    setBusy(true); setError(""); setSplitRecovery(false);
    try {
      const selectedDevice = device === "cuda" ? (gpuIds.join(",") || "0") : device;
      const task = await api.createTrainingTask({ dataset_id: dataset.id, name: name.trim() || "local-training", model, task_type: dataset.task_type as TaskType, epochs, img_size: imgSize, batch_size: batchSize, device: selectedDevice, workers, seed: 42 });
      setTasks((items) => [task, ...items]);
      setSelectedTask(task);
    } catch (reason) {
      const message = errorMessage(reason);
      setError(message);
      setSplitRecovery(message.includes("at least one train image and one val image"));
    } finally { setBusy(false); }
  };

  const autoSplitForTraining = async () => {
    setBusy(true); setError("");
    try {
      const result = await api.autoSplitImages(dataset.id, { train_ratio: 0.8, val_ratio: 0.2, test_ratio: 0, seed: 42 });
      setSplitRecovery(false);
      setError(`已按 80/20 分配 ${result.updated} 张图片：train ${result.split_counts.train}，val ${result.split_counts.val}。现在可以重新创建训练。`);
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const stop = async () => {
    if (!selectedTask) return;
    setBusy(true);
    try { setSelectedTask(await api.stopTrainingTask(selectedTask.id)); await refresh(); } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const resume = async () => {
    if (!selectedTask) return;
    setBusy(true); setError("");
    try {
      const restoreEpochState = selectedTask.status !== "completed";
      const task = await api.resumeTrainingTask(selectedTask.id, restoreEpochState ? { resume_epoch: true } : { epochs, resume_epoch: false });
      setTasks((items) => [task, ...items]); setSelectedTask(task);
    }
    catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const checkpointLinks = useMemo(() => selectedTask ? { best: api.downloadCheckpointUrl(selectedTask.id, "best"), last: api.downloadCheckpointUrl(selectedTask.id, "last") } : undefined, [selectedTask]);
  return (
    <main className="training-layout">
      <div className="training-head">
        <button className="button" onClick={onBack}>← 返回数据集</button>
        <div><span className="eyebrow">LOCAL TRAINING / {dataset.task_type}</span><h1>{dataset.name}</h1><p className="muted">复用已保存的 train / val / test split，在本机运行 Ultralytics YOLO。</p></div>
      </div>
      {error && <div className={`validation ${splitRecovery ? "invalid" : "valid"}`}><strong>{splitRecovery ? "训练尚未创建" : "训练数据已准备"}</strong><span>{error}</span>{splitRecovery && <div className="training-split-recovery"><small>这会重新分配该数据集全部图片的持久化 split；请确认当前 split 不需要保留。</small><button className="button" disabled={busy || dataset.image_count < 2} onClick={() => void autoSplitForTraining()}>自动分配 train / val</button>{dataset.image_count < 2 && <small>至少需要两张图片，才能同时创建 train 与 val split。</small>}</div>}</div>}
      <div className="training-grid">
        <section className="panel training-form">
          <span className="eyebrow">NEW TRAINING TASK</span><h2>训练配置</h2>
          <p className="hint">模型权重族必须和数据集任务匹配；首次运行可能需要下载 Ultralytics 权重。</p>
          <label>训练数据集<select value={dataset.id} onChange={(event) => { const next = datasets.find((item) => item.id === event.target.value); if (next) onDatasetChange(next); }}>{datasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.task_type} · {item.image_count} 张图片</option>)}</select></label>
          <label>任务名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>基础权重<input value={model} onChange={(event) => setModel(event.target.value)} /><small>建议使用 {defaultModelFor(dataset.task_type)}</small></label>
          <div className="form-row"><label>Epochs<input type="number" min={1} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label><label>Batch<input type="number" min={1} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label></div>
          <div className="form-row"><label>Image size<input type="number" min={32} step={32} value={imgSize} onChange={(event) => setImgSize(Number(event.target.value))} /></label><label>Workers<input type="number" min={0} value={workers} onChange={(event) => setWorkers(Number(event.target.value))} /></label></div>
          <label>Device<select value={device} onChange={(event) => { const next = event.target.value; setDevice(next); if (next === "cuda" && !gpuIds.length) { const first = devices.find((item) => item.type === "cuda"); if (first?.index !== null && first?.index !== undefined) setGpuIds([String(first.index)]); } }}><option value="auto">auto（CUDA → MPS → CPU）</option><option value="cpu">cpu</option>{devices.some((item) => item.type === "mps") && <option value="mps">mps</option>}{devices.some((item) => item.type === "cuda") && <option value="cuda">CUDA</option>}</select></label>
          {device === "cuda" && <GpuPicker devices={devices} mode={gpuMode} selectedIds={gpuIds} onModeChange={setGpuMode} onIdsChange={setGpuIds} />}
          <button className="button primary wide" disabled={busy || (device === "cuda" && (!gpuIds.length || (gpuMode === "multi" && gpuIds.length < 2)))} onClick={start}>创建并运行训练</button>
        </section>
        <section className="panel task-panel"><div className="panel-heading"><div><span className="eyebrow">TASK QUEUE</span><h2>训练任务</h2></div><span className="count-badge">{tasks.length}</span></div><div className="task-list">{tasks.map((task) => <button key={task.id} className={selectedTask?.id === task.id ? "task-item selected" : "task-item"} onClick={() => setSelectedTask(task)}><div><strong>{task.name}</strong><small>{task.model_name} · {task.created_at.slice(0, 16).replace("T", " ")}</small></div><span className={`status ${task.status}`}>{statusLabel(task.status)}</span><div className="progress-line"><i style={{ width: `${task.progress_percent}%` }} /></div></button>)}{!tasks.length && <p className="muted">还没有训练任务。</p>}</div></section>
      </div>
      {selectedTask && <section className="panel training-detail">
        <div className="detail-head"><div><span className="eyebrow">TASK DETAIL</span><h2>{selectedTask.name}</h2><p className="muted">{selectedTask.status} · {selectedTask.progress_epoch}/{selectedTask.progress_total_epochs || selectedTask.epochs} epochs · {selectedTask.progress_percent}%</p></div><div className="detail-actions">{selectedTask.status === "running" || selectedTask.status === "pending" ? <button className="button danger" disabled={busy} onClick={stop}>停止训练</button> : <><button className="button primary" disabled={busy || !selectedTask.last_model_path} onClick={resume}>{selectedTask.status === "completed" ? "从 last.pt 新任务继续训练" : "恢复中断训练"}</button>{checkpointLinks?.best && selectedTask.best_model_path && <a className="button" href={checkpointLinks.best}>下载 best.pt</a>}{checkpointLinks?.last && selectedTask.last_model_path && <a className="button" href={checkpointLinks.last}>下载 last.pt</a>}</>}</div></div>
        {selectedTask.error_message && <div className="validation invalid"><span>{selectedTask.error_message}</span></div>}
        <TrainingSummaryPanel summary={summary} />
        <details className="config-snapshot"><summary>训练配置快照</summary><pre>{JSON.stringify(summary?.training_config ?? { model: selectedTask.model_name, epochs: selectedTask.epochs, img_size: selectedTask.img_size, batch_size: selectedTask.batch_size, device: selectedTask.device, workers: selectedTask.workers }, null, 2)}</pre></details>
        <div className="log-box">{logs?.logs || "等待训练日志…"}</div>
      </section>}
    </main>
  );
}

function statusLabel(status: TrainingTask["status"]): string { return ({ pending: "排队中", running: "运行中", completed: "已完成", failed: "失败", stopped: "已停止" })[status]; }
function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
function defaultModelFor(taskType: TaskType): string { return ({ detect: "yolo11n.pt", segment: "yolo11n-seg.pt", obb: "yolo11n-obb.pt", classify: "yolo11n-cls.pt" })[taskType]; }

function GpuPicker({ devices, mode, selectedIds, onModeChange, onIdsChange }: { devices: TrainingDevice[]; mode: "single" | "multi"; selectedIds: string[]; onModeChange: (mode: "single" | "multi") => void; onIdsChange: (ids: string[]) => void }) {
  const cudaDevices = devices.filter((item) => item.type === "cuda");
  const toggle = (id: string) => {
    if (mode === "single") { onIdsChange([id]); return; }
    onIdsChange(selectedIds.includes(id) ? selectedIds.filter((item) => item !== id) : [...selectedIds, id]);
  };
  return <div className="gpu-picker">
    <div className="gpu-mode-toggle"><button type="button" className={mode === "single" ? "active" : ""} onClick={() => { onModeChange("single"); onIdsChange(selectedIds.slice(0, 1)); }}>单 GPU</button><button type="button" className={mode === "multi" ? "active" : ""} onClick={() => onModeChange("multi")}>多 GPU</button></div>
    <div className="gpu-list">
      {cudaDevices.map((gpu) => {
        const id = String(gpu.index ?? 0);
        const checked = selectedIds.includes(id);
        return (
          <label key={gpu.id} className={checked ? "gpu-option selected" : "gpu-option"}>
            <input type="checkbox" checked={checked} disabled={mode === "single" && checked} onChange={() => toggle(id)} />
            <span><strong>GPU {id}</strong><small>{gpu.name}{gpu.memory_total_mb ? ` · ${Math.round(gpu.memory_total_mb / 1024)} GB` : ""}</small></span>
          </label>
        );
      })}
    </div>
    <small>{"多 GPU 将以 device="}{selectedIds.join(",") || "0,1"}{" 交给 Ultralytics DDP。"}</small>
  </div>;
}

function TrainingSummaryPanel({ summary }: { summary?: TrainingSummary }) {
  const value = (key: string) => typeof summary?.metrics[key] === "number" ? Number(summary.metrics[key]).toFixed(3) : "—";
  const history = summary?.metrics.history ?? [];
  const path = history.map((point, index) => `${index ? "L" : "M"}${index * (100 / Math.max(history.length - 1, 1))} ${72 - (point.map50 ?? 0) * 64}`).join(" ");
  return <div className="training-result"><div className="metric-strip"><Metric label="mAP50" value={value("map50")} /><Metric label="mAP50-95" value={value("map50_95")} /><Metric label="Precision" value={value("precision")} /><Metric label="Recall" value={value("recall")} /></div>{history.length > 1 && <div className="metric-chart"><div><strong>mAP50 趋势</strong><small>{history.length} 个训练轮次</small></div><svg viewBox="0 0 100 80" preserveAspectRatio="none"><path d="M0 72H100" stroke="#dbe4ef" /><path d={path} fill="none" stroke="#3157d5" strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg></div>}{summary?.risks.length ? <p className="hint">风险提示：{summary.risks.join("、")}</p> : null}</div>;
}
function Metric({ label, value }: { label: string; value: string }) { return <span><small>{label}</small><strong>{value}</strong></span>; }
