import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { AutoAnnotationLog, AutoAnnotationTask, ClassLabel, Dataset, ModelVersion } from "../types";

const terminalStatuses = new Set(["completed", "failed", "stopped"]);

export function AutoAnnotationModal({ dataset, onClose, onChanged }: { dataset: Dataset; onClose: () => void; onChanged: () => void }) {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [modelId, setModelId] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [cleanOld, setCleanOld] = useState(false);
  const [confirmCleanOld, setConfirmCleanOld] = useState(false);
  const [targetClasses, setTargetClasses] = useState<ClassLabel[]>([]);
  const [sourceClasses, setSourceClasses] = useState<ClassLabel[]>([]);
  const [classMapping, setClassMapping] = useState<Record<string, string>>({});
  const [classesLoading, setClassesLoading] = useState(true);
  const [task, setTask] = useState<AutoAnnotationTask>();
  const [log, setLog] = useState<AutoAnnotationLog>();
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [changedTaskId, setChangedTaskId] = useState("");

  useEffect(() => {
    Promise.all([api.listModels(undefined, false), api.listClasses(dataset.id)])
      .then(([result, target]) => {
        const available = result.items.filter((item) => item.status === "active" && item.format === "pt" && item.task_type === dataset.task_type && item.dataset_id);
        setModels(available);
        setModelId(available[0]?.id ?? "");
        setTargetClasses(target);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "读取可用模型失败"))
      .finally(() => { setLoading(false); setClassesLoading(false); });
  }, [dataset.id, dataset.task_type]);

  const selectedModel = useMemo(() => models.find((item) => item.id === modelId), [models, modelId]);

  useEffect(() => {
    if (!selectedModel?.dataset_id) {
      setSourceClasses([]);
      setClassMapping({});
      return;
    }
    let active = true;
    setClassesLoading(true);
    api.listClasses(selectedModel.dataset_id)
      .then((source) => {
        if (!active) return;
        setSourceClasses(source);
        const targetByName = new Map(targetClasses.map((item) => [item.name.trim().toLocaleLowerCase(), item.id]));
        setClassMapping(Object.fromEntries(source.map((item) => [String(item.class_index), targetByName.get(item.name.trim().toLocaleLowerCase()) ?? ""])));
      })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "读取模型类别失败"); })
      .finally(() => { if (active) setClassesLoading(false); });
    return () => { active = false; };
  }, [selectedModel?.dataset_id, targetClasses]);

  useEffect(() => {
    if (!task || terminalStatuses.has(task.status)) return;
    let active = true;
    const refresh = async () => {
      try {
        const next = await api.getAutoAnnotation(task.id);
        if (!active) return;
        setTask(next);
        setLog(await api.autoAnnotationLogs(next.id));
        if (terminalStatuses.has(next.status)) setChangedTaskId(next.id);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "读取自动标注进度失败");
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => { active = false; window.clearInterval(timer); };
  }, [task?.id, task?.status]);

  useEffect(() => {
    if (!changedTaskId) return;
    onChanged();
    setChangedTaskId("");
  }, [changedTaskId, onChanged]);

  const mappedClassCount = Object.values(classMapping).filter(Boolean).length;
  const start = async () => {
    if (!modelId) return;
    setBusy(true);
    setError("");
    try {
      const mapping = Object.fromEntries(Object.entries(classMapping).filter(([, targetId]) => Boolean(targetId)));
      const created = await api.createAutoAnnotation(dataset.id, { model_id: modelId, confidence, iou, clean_old_annotations: cleanOld, class_mapping: mapping });
      setTask(created);
      setLog(undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "启动自动标注失败");
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    if (!task) return;
    setBusy(true);
    try { setTask(await api.stopAutoAnnotation(task.id)); } catch (reason) { setError(reason instanceof Error ? reason.message : "停止自动标注失败"); } finally { setBusy(false); }
  };
  const running = !!task && !terminalStatuses.has(task.status);

  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="dataset-dialog auto-annotation-dialog" role="dialog" aria-modal="true" aria-labelledby="auto-annotation-title" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">AUTO ANNOTATION</span><h2 id="auto-annotation-title">自动标注“{dataset.name}”</h2><p>选择一个受管模型生成初始标注，完成后仍可在标注页逐张审核、修改或删除。</p></div><button className="icon-button" onClick={onClose} aria-label="关闭">×</button></header>
      {!task && <>
        {loading ? <p className="auto-annotation-empty">正在读取可用模型…</p> : models.length ? <>
          <label>模型<select value={modelId} onChange={(event) => setModelId(event.target.value)}>{models.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.artifact_type.toUpperCase()}</option>)}</select></label>
          {selectedModel && <p className="hint auto-annotation-model-hint">任务类型：{selectedModel.task_type} · 仅使用本地受管 PT 模型。同名类别会自动映射，其余类别需明确选择目标类别或忽略。</p>}
          <section className="auto-annotation-mapping"><div><strong>类别映射</strong><small>未映射的模型类别会跳过，绝不会按类别索引猜测。</small></div>{classesLoading ? <p className="hint">正在读取模型类别…</p> : sourceClasses.map((source) => <label key={source.id}><span>{source.class_index}: {source.name}</span><select value={classMapping[String(source.class_index)] ?? ""} onChange={(event) => setClassMapping((current) => ({ ...current, [String(source.class_index)]: event.target.value }))}><option value="">忽略此类别</option>{targetClasses.map((target) => <option key={target.id} value={target.id}>{target.class_index}: {target.name}</option>)}</select></label>)}</section>
          <div className="auto-annotation-controls"><label>置信度 <strong>{confidence.toFixed(2)}</strong><input type="range" min="0.01" max="0.99" step="0.01" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /></label><label>IoU <strong>{iou.toFixed(2)}</strong><input type="range" min="0" max="0.99" step="0.01" value={iou} onChange={(event) => setIou(Number(event.target.value))} /></label></div>
          <label className="auto-annotation-checkbox"><input type="checkbox" checked={cleanOld} onChange={(event) => { setCleanOld(event.target.checked); setConfirmCleanOld(false); }} /><span><strong>清理旧标注</strong><small>关闭时保留人工/导入标注，只替换上一次自动标注结果。</small></span></label>
          {cleanOld && <label className="auto-annotation-checkbox danger"><input type="checkbox" checked={confirmCleanOld} onChange={(event) => setConfirmCleanOld(event.target.checked)} /><span><strong>我理解这会删除每张图片现有的全部标注</strong><small>包括人工、导入和先前自动生成的标注，且无法撤销。</small></span></label>}
        </> : <p className="auto-annotation-empty">当前没有匹配的受管 PT 模型。请先完成一次本地训练并登记模型。</p>}
        {error && <div className="validation invalid"><span>{error}</span></div>}
        <footer><button className="button" onClick={onClose}>取消</button><button className="button primary" disabled={busy || loading || classesLoading || !modelId || !mappedClassCount || (cleanOld && !confirmCleanOld)} onClick={() => void start()}>{busy ? "正在启动…" : "开始自动标注"}</button></footer>
      </>}
      {task && <>
        <div className="auto-annotation-status"><div><span className={`auto-annotation-status-dot ${task.status}`} /><strong>{statusLabel(task.status)}</strong><span>{task.processed_images} / {task.total_images} 张图片</span></div><b>{Math.round(task.progress_percent)}%</b></div>
        <div className="progress-track"><i style={{ width: `${task.progress_percent}%` }} /></div>
        <div className="auto-annotation-summary"><span><small>生成标注</small><strong>{task.created_annotations}</strong></span><span><small>跳过图片</small><strong>{task.skipped_images}</strong></span><span><small>置信度</small><strong>{task.confidence.toFixed(2)}</strong></span></div>
        {task.error_message && <div className={task.status === "failed" ? "validation invalid" : "validation"}><span>{task.error_message}</span></div>}
        {log?.logs && <pre className="auto-annotation-log">{log.logs}</pre>}
        <footer>{running ? <button className="button danger" disabled={busy || task.stop_requested} onClick={() => void stop()}>{task.stop_requested ? "正在停止…" : "取消任务"}</button> : <span className="hint">结果已写入数据集，请继续人工审核。</span>}<button className="button" onClick={onClose}>{running ? "后台运行" : "完成"}</button></footer>
      </>}
    </section>
  </div>;
}

function statusLabel(status: AutoAnnotationTask["status"]): string {
  return ({ pending: "排队中", running: "正在标注", completed: "已完成", failed: "执行失败", stopped: "已取消" })[status];
}
