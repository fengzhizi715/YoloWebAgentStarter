import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Dataset, TaskType, TrainingLog, TrainingTask } from "../types";

interface Props {
  dataset: Dataset;
  onBack: () => void;
}

export function TrainingView({ dataset, onBack }: Props) {
  const defaultModel = defaultModelFor(dataset.task_type);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<TrainingTask>();
  const [logs, setLogs] = useState<TrainingLog>();
  const [name, setName] = useState("local-training");
  const [model, setModel] = useState(defaultModel);
  const [epochs, setEpochs] = useState(50);
  const [imgSize, setImgSize] = useState(640);
  const [batchSize, setBatchSize] = useState(16);
  const [device, setDevice] = useState("auto");
  const [workers, setWorkers] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const result = await api.listTrainingTasks(dataset.id);
    setTasks(result.items);
    setSelectedTask((current) => result.items.find((item) => item.id === current?.id) ?? result.items[0]);
  }, [dataset.id]);

  useEffect(() => {
    refresh().catch((reason) => setError(errorMessage(reason)));
  }, [refresh]);

  useEffect(() => {
    const active = selectedTask && (selectedTask.status === "pending" || selectedTask.status === "running");
    if (!active) return;
    const timer = window.setInterval(() => refresh().catch((reason) => setError(errorMessage(reason))), 1500);
    return () => window.clearInterval(timer);
  }, [refresh, selectedTask]);

  useEffect(() => {
    if (!selectedTask) { setLogs(undefined); return; }
    api.getTrainingLogs(selectedTask.id).then(setLogs).catch((reason) => setError(errorMessage(reason)));
  }, [selectedTask]);

  const start = async () => {
    setBusy(true); setError("");
    try {
      const task = await api.createTrainingTask({ dataset_id: dataset.id, name: name.trim() || "local-training", model, task_type: dataset.task_type as TaskType, epochs, img_size: imgSize, batch_size: batchSize, device, workers, seed: 42 });
      setTasks((items) => [task, ...items]);
      setSelectedTask(task);
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const stop = async () => {
    if (!selectedTask) return;
    setBusy(true);
    try { setSelectedTask(await api.stopTrainingTask(selectedTask.id)); await refresh(); } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const checkpointLinks = useMemo(() => selectedTask ? { best: api.downloadCheckpointUrl(selectedTask.id, "best"), last: api.downloadCheckpointUrl(selectedTask.id, "last") } : undefined, [selectedTask]);
  return (
    <main className="training-layout">
      <div className="training-head"><button className="button" onClick={onBack}>← 返回数据集</button><div><span className="eyebrow">LOCAL TRAINING / {dataset.task_type}</span><h1>{dataset.name}</h1><p className="muted">复用已保存的 train / val / test split，在本机运行 Ultralytics YOLO。</p></div></div>
      {error && <div className="validation invalid"><strong>训练操作失败</strong><span>{error}</span></div>}
      <div className="training-grid">
        <section className="panel training-form"><span className="eyebrow">NEW TRAINING TASK</span><h2>训练配置</h2><p className="hint">模型权重族必须和数据集任务匹配；首次运行可能需要下载 Ultralytics 权重。</p><label>任务名称<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>基础权重<input value={model} onChange={(event) => setModel(event.target.value)} /><small>建议使用 {defaultModelFor(dataset.task_type)}</small></label><div className="form-row"><label>Epochs<input type="number" min={1} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label><label>Batch<input type="number" min={1} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label></div><div className="form-row"><label>Image size<input type="number" min={32} step={32} value={imgSize} onChange={(event) => setImgSize(Number(event.target.value))} /></label><label>Workers<input type="number" min={0} value={workers} onChange={(event) => setWorkers(Number(event.target.value))} /></label></div><label>Device<select value={device} onChange={(event) => setDevice(event.target.value)}><option value="auto">auto</option><option value="cpu">cpu</option><option value="mps">mps</option><option value="0">cuda:0</option></select></label><button className="button primary wide" disabled={busy} onClick={start}>创建并运行训练</button></section>
        <section className="panel task-panel"><div className="panel-heading"><div><span className="eyebrow">TASK QUEUE</span><h2>训练任务</h2></div><span className="count-badge">{tasks.length}</span></div><div className="task-list">{tasks.map((task) => <button key={task.id} className={selectedTask?.id === task.id ? "task-item selected" : "task-item"} onClick={() => setSelectedTask(task)}><div><strong>{task.name}</strong><small>{task.model_name} · {task.created_at.slice(0, 16).replace("T", " ")}</small></div><span className={`status ${task.status}`}>{statusLabel(task.status)}</span><div className="progress-line"><i style={{ width: `${task.progress_percent}%` }} /></div></button>)}{!tasks.length && <p className="muted">还没有训练任务。</p>}</div></section>
      </div>
      {selectedTask && <section className="panel training-detail"><div className="detail-head"><div><span className="eyebrow">TASK DETAIL</span><h2>{selectedTask.name}</h2><p className="muted">{selectedTask.status} · {selectedTask.progress_epoch}/{selectedTask.progress_total_epochs || selectedTask.epochs} epochs · {selectedTask.progress_percent}%</p></div><div className="detail-actions">{selectedTask.status === "running" || selectedTask.status === "pending" ? <button className="button danger" disabled={busy} onClick={stop}>停止训练</button> : <>{checkpointLinks?.best && selectedTask.best_model_path && <a className="button" href={checkpointLinks.best}>下载 best.pt</a>}{checkpointLinks?.last && selectedTask.last_model_path && <a className="button" href={checkpointLinks.last}>下载 last.pt</a>}</>}</div></div>{selectedTask.error_message && <div className="validation invalid"><span>{selectedTask.error_message}</span></div>}<div className="log-box">{logs?.logs || "等待训练日志…"}</div></section>}
    </main>
  );
}

function statusLabel(status: TrainingTask["status"]): string { return ({ pending: "排队中", running: "运行中", completed: "已完成", failed: "失败", stopped: "已停止" })[status]; }
function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
function defaultModelFor(taskType: TaskType): string { return ({ detect: "yolo11n.pt", segment: "yolo11n-seg.pt", obb: "yolo11n-obb.pt", classify: "yolo11n-cls.pt" })[taskType]; }
