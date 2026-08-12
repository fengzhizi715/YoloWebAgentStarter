import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Dataset, ImageItem, ModelComparison, ModelEvaluationRecord, ModelTestRecord, ModelVersion, SplitName } from "../types";
import { ModelTestModal } from "../components/ModelTestModal";
import { ModelEvaluationPanel } from "../components/ModelEvaluationPanel";

interface Props {
  dataset: Dataset;
  onBack: () => void;
  onOpenPreannotated: (image: ImageItem, drafts: unknown[]) => void;
}

export function ModelsView({ dataset, onBack, onOpenPreannotated }: Props) {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [selected, setSelected] = useState<ModelVersion>();
  const [name, setName] = useState("");
  const [version, setVersion] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [testing, setTesting] = useState<ModelVersion>();
  const [images, setImages] = useState<ImageItem[]>([]);
  const [compareWith, setCompareWith] = useState("");
  const [comparison, setComparison] = useState<ModelComparison>();
  const [preannotationImageIds, setPreannotationImageIds] = useState<string[]>([]);
  const [preannotationResults, setPreannotationResults] = useState<Array<{ image_id: string; annotations: unknown[] }>>([]);
  const [testRecords, setTestRecords] = useState<ModelTestRecord[]>([]);
  const [evaluations, setEvaluations] = useState<ModelEvaluationRecord[]>([]);

  const refresh = useCallback(async () => {
    const result = await api.listModels(dataset.id);
    setModels(result.items);
    setSelected((current) => result.items.find((item) => item.id === current?.id) ?? result.items[0]);
  }, [dataset.id]);

  useEffect(() => {
    refresh().catch((reason) => setError(errorMessage(reason)));
    api.listImages(dataset.id).then((result) => setImages(result.items)).catch((reason) => setError(errorMessage(reason)));
  }, [refresh]);

  useEffect(() => {
    setName(selected?.name ?? "");
    setVersion(selected?.version ?? "");
    setNotes(selected?.notes ?? "");
  }, [selected]);

  useEffect(() => {
    if (!selected) { setTestRecords([]); setEvaluations([]); return; }
    api.listModelTests(selected.id).then(setTestRecords).catch((reason) => setError(errorMessage(reason)));
    api.listModelEvaluations(selected.id).then(setEvaluations).catch((reason) => setError(errorMessage(reason)));
  }, [selected?.id]);

  useEffect(() => {
    if (!selected || !evaluations.some((item) => item.status === "pending" || item.status === "running")) return;
    const timer = window.setInterval(() => api.listModelEvaluations(selected.id).then(setEvaluations).catch((reason) => setError(errorMessage(reason))), 1500);
    return () => window.clearInterval(timer);
  }, [evaluations, selected?.id]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try { await action(); } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const save = () => selected && run(async () => {
    const updated = await api.updateModel(selected.id, { name: name.trim(), version: version.trim(), notes });
    setSelected(updated);
    await refresh();
  });

  const exportOnnx = () => selected && run(async () => {
    const exported = await api.exportModelOnnx(selected.id);
    await refresh();
    setSelected(exported);
  });

  const toggleArchive = () => selected && run(async () => {
    const updated = selected.status === "archived" ? await api.restoreModel(selected.id) : await api.archiveModel(selected.id);
    setSelected(updated);
    await refresh();
  });

  const remove = () => selected && window.confirm(`删除模型 ${selected.name}？`) && run(async () => {
    await api.deleteModel(selected.id);
    await refresh();
  });
  const compare = () => selected && compareWith && run(async () => setComparison(await api.compareModels(selected.id, compareWith)));
  const openPreannotation = (proposal: { image_id: string; annotations: unknown[] }) => { const image = images.find((item) => item.id === proposal.image_id); if (image) onOpenPreannotated(image, proposal.annotations); };
  const preannotate = () => selected && preannotationImageIds.length && run(async () => { const result = await api.preannotate(selected.id, dataset.id, preannotationImageIds); setPreannotationResults(result.images); if (result.images[0]) openPreannotation(result.images[0]); });
  const evaluate = (split: SplitName) => selected && run(async () => { const record = await api.evaluateModel(selected.id, split); setEvaluations((items) => [record, ...items]); });

  return (
    <main className="models-layout">
      <div className="models-head"><button className="button" onClick={onBack}>← 返回数据集</button><div><span className="eyebrow">MODEL REGISTRY / {dataset.task_type}</span><h1>{dataset.name}</h1><p className="muted">管理训练产生的 PT，并从 PT 生成 ONNX FP32。</p></div></div>
      {error && <div className="validation invalid"><strong>模型操作失败</strong><span>{error}</span></div>}
      <div className="models-grid">
        <section className="panel model-list-panel"><div className="panel-heading"><div><span className="eyebrow">MODEL VERSIONS</span><h2>模型列表</h2></div><span className="count-badge">{models.length}</span></div><div className="model-list">{models.map((model) => <button key={model.id} className={selected?.id === model.id ? "model-item selected" : "model-item"} onClick={() => setSelected(model)}><div><strong>{model.name}</strong><small>{model.format.toUpperCase()} · {model.artifact_type} · {model.version}</small></div><span className={`status ${model.status}`}>{model.status === "active" ? "启用" : "已归档"}</span></button>)}{!models.length && <div className="empty-state"><strong>暂无模型</strong><span>完成训练后，best.pt 和 last.pt 会自动进入模型列表。</span></div>}</div></section>
      {selected ? <section className="panel model-detail-panel"><div className="detail-head"><div><span className="eyebrow">MODEL DETAIL</span><h2>{selected.name}</h2><p className="muted">{selected.format.toUpperCase()} · {selected.task_type} · {selected.source}</p></div><span className={`status ${selected.status}`}>{selected.status === "active" ? "启用" : "已归档"}</span></div><div className="model-actions"><a className="button" href={api.downloadModelUrl(selected.id)}>下载 {selected.format.toUpperCase()}</a>{selected.format === "pt" && <><button className="button primary" disabled={busy} onClick={() => setTesting(selected)}>快速测试</button><button className="button" disabled={busy} onClick={exportOnnx}>导出 ONNX FP32</button></>}<button className="button" disabled={busy} onClick={toggleArchive}>{selected.status === "active" ? "归档" : "恢复"}</button><button className="button danger" disabled={busy} onClick={remove}>删除</button></div>{selected.format === "pt" && <div className="model-tools"><div><span className="eyebrow">REVIEWABLE PRE-ANNOTATION</span><h3>批量生成后逐张确认</h3><p>可多选至多 100 张；建议不会直接保存为标签。</p><select multiple value={preannotationImageIds} onChange={(event) => setPreannotationImageIds(Array.from(event.currentTarget.selectedOptions, (option) => option.value))}>{images.map((image) => <option key={image.id} value={image.id}>{image.file_name}</option>)}</select><button className="button" disabled={busy || !preannotationImageIds.length} onClick={preannotate}>生成并打开首张审阅</button>{preannotationResults.length > 0 && <div className="comparison"><strong>{preannotationResults.length} 张建议待确认</strong>{preannotationResults.map((proposal) => <button className="button" key={proposal.image_id} onClick={() => openPreannotation(proposal)}>审阅 {images.find((item) => item.id === proposal.image_id)?.file_name ?? proposal.image_id}</button>)}</div>}</div><div><span className="eyebrow">MODEL COMPARISON</span><h3>与同数据集模型对比</h3><select value={compareWith} onChange={(event) => setCompareWith(event.target.value)}><option value="">选择候选模型</option>{models.filter((item) => item.id !== selected.id && item.task_type === selected.task_type).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><button className="button" disabled={busy || !compareWith} onClick={compare}>比较指标</button>{comparison && <div className="comparison"><strong>{comparison.candidate.name} vs {comparison.baseline.name}</strong>{Object.entries(comparison.deltas).map(([key, value]) => <span key={key}>{key}: {value == null ? "—" : value.toFixed(3)}</span>)}<small>{comparison.suggestions[0]}</small></div>}</div><ModelEvaluationPanel model={selected} evaluations={evaluations} busy={busy} onEvaluate={evaluate} /></div>}<div className="model-form"><label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>版本<input value={version} onChange={(event) => setVersion(event.target.value)} /></label><label>备注<textarea value={notes} onChange={(event) => setNotes(event.target.value)} /></label><button className="button" disabled={busy || !name.trim() || !version.trim()} onClick={save}>保存元数据</button></div><div className="model-meta"><span>文件：{selected.model_path}</span><span>基础模型：{selected.base_model || "—"}</span><span>mAP50：{selected.map50 ?? "—"}</span><span>测试记录：{testRecords.length}</span><span>创建：{selected.created_at.slice(0, 16).replace("T", " ")}</span></div></section> : <section className="panel model-detail-panel empty-state"><strong>选择一个模型查看详情</strong></section>}
      </div>
      {testing && <ModelTestModal model={testing} onClose={() => setTesting(undefined)} />}
    </main>
  );
}

function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
