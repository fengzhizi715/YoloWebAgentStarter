import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Dataset, ModelComparison, ModelEvaluationRecord, ModelVersion, SplitName } from "../types";
import { ModelTestModal } from "../components/ModelTestModal";
import { ModelComparisonBar } from "../components/model-list/ModelComparisonBar";
import { ModelLibraryCard } from "../components/model-list/ModelLibraryCard";
import { ModelLibraryHero } from "../components/model-list/ModelLibraryHero";
import { ModelLibraryStats } from "../components/model-list/ModelLibraryStats";
import { ModelDetailView } from "../components/model-detail/ModelDetailView";
import { formatDelta, filterModels, formatMetric, newerFirst, pickBestModel, sortModels, type ModelListFilter, type ModelListSort } from "../models/helpers";
import { taskTypeLabel } from "../training/helpers";
import { IconSearch } from "../components/training/icons";

interface Props {
  dataset: Dataset;
}

export function ModelsView({ dataset }: Props) {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [detailId, setDetailId] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [compareBusy, setCompareBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [testing, setTesting] = useState<ModelVersion>();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ModelComparison>();
  const [evaluations, setEvaluations] = useState<ModelEvaluationRecord[]>([]);
  const [includeArchived, setIncludeArchived] = useState(true);
  const [search, setSearch] = useState("");
  const [listFilter, setListFilter] = useState<ModelListFilter>("all");
  const [listSort, setListSort] = useState<ModelListSort>("newest");

  const refresh = useCallback(async () => {
    const result = await api.listModels(dataset.id, includeArchived);
    const items = [...result.items].sort(newerFirst);
    setModels(items);
    setDetailId((current) => (current && items.some((item) => item.id === current) ? current : undefined));
    setSelectedIds((ids) => ids.filter((id) => items.some((item) => item.id === id)));
  }, [dataset.id, includeArchived]);

  useEffect(() => {
    refresh().catch((reason) => setError(errorMessage(reason)));
  }, [refresh]);

  const detailModel = useMemo(() => models.find((item) => item.id === detailId), [models, detailId]);
  const bestModel = useMemo(() => pickBestModel(models.filter((item) => item.status === "active")), [models]);
  const filteredModels = useMemo(
    () => sortModels(filterModels(models, listFilter, search), listSort),
    [models, listFilter, search, listSort],
  );
  const filterCounts = useMemo(() => ({
    all: models.length,
    pt: models.filter((item) => item.format === "pt").length,
    onnx: models.filter((item) => item.format === "onnx").length,
    best: models.filter((item) => item.artifact_type === "best").length,
    archived: models.filter((item) => item.status === "archived").length,
  }), [models]);

  useEffect(() => {
    if (!detailModel || detailModel.format !== "pt") { setEvaluations([]); return; }
    api.listModelEvaluations(detailModel.id).then(setEvaluations).catch((reason) => setError(errorMessage(reason)));
  }, [detailModel?.id, detailModel?.format]);

  useEffect(() => {
    if (!detailModel || !evaluations.some((item) => item.status === "pending" || item.status === "running")) return;
    const timer = window.setInterval(() => {
      api.listModelEvaluations(detailModel.id).then(setEvaluations).catch((reason) => setError(errorMessage(reason)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [evaluations, detailModel?.id]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    setNotice("");
    try { await action(); } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const exportOnnx = (model: ModelVersion) => run(async () => {
    const exported = await api.exportModelOnnx(model.id);
    await refresh();
    setDetailId(exported.id);
    setNotice(`已导出 ONNX：${exported.name}`);
  });

  const toggleArchive = (model: ModelVersion) => run(async () => {
    const updated = model.status === "archived" ? await api.restoreModel(model.id) : await api.archiveModel(model.id);
    await refresh();
    setDetailId(updated.id);
  });

  const remove = (model: ModelVersion) => {
    if (!window.confirm(`删除模型 ${model.name}？`)) return;
    void run(async () => {
      await api.deleteModel(model.id);
      if (detailId === model.id) setDetailId(undefined);
      await refresh();
    });
  };

  const removeSelected = () => {
    if (!selectedIds.length) return;
    if (!window.confirm(`删除选中的 ${selectedIds.length} 个模型？`)) return;
    void run(async () => {
      for (const id of selectedIds) await api.deleteModel(id);
      setSelectedIds([]);
      setComparison(undefined);
      if (detailId && selectedIds.includes(detailId)) setDetailId(undefined);
      await refresh();
    });
  };

  const compareSelected = async () => {
    if (selectedIds.length !== 2) return;
    setCompareBusy(true);
    setError("");
    try {
      setComparison(await api.compareModels(selectedIds[0], selectedIds[1]));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setCompareBusy(false);
    }
  };

  const compareWithBest = async () => {
    if (!bestModel || !selectedIds.length) return;
    const candidate = selectedIds.find((id) => id !== bestModel.id) ?? selectedIds[0];
    if (candidate === bestModel.id) {
      setError("请选择与最佳模型不同的候选模型。");
      return;
    }
    setCompareBusy(true);
    setError("");
    try {
      setComparison(await api.compareModels(bestModel.id, candidate));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setCompareBusy(false);
    }
  };

  const evaluate = (split: SplitName) => detailModel && run(async () => {
    const record = await api.evaluateModel(detailModel.id, split);
    setEvaluations((items) => [record, ...items]);
  });

  if (detailModel) {
    return (
      <main className="models-layout models-workspace-page">
        {error && <div className="validation invalid"><strong>模型操作失败</strong><span>{error}</span></div>}
        {notice && <p className="training-workspace-notice">{notice}</p>}
        <ModelDetailView
          dataset={dataset}
          model={detailModel}
          models={models}
          evaluations={evaluations}
          busy={busy}
          onBack={() => setDetailId(undefined)}
          onSelectModel={(model) => setDetailId(model.id)}
          onTest={setTesting}
          onExportOnnx={() => void exportOnnx(detailModel)}
          onToggleArchive={() => void toggleArchive(detailModel)}
          onDelete={() => remove(detailModel)}
          onEvaluate={evaluate}
          onSaved={async (updated) => {
            setDetailId(updated.id);
            await refresh();
            setNotice("元数据已保存");
          }}
        />
        {testing && <ModelTestModal model={testing} onClose={() => setTesting(undefined)} />}
      </main>
    );
  }

  return (
    <main className="models-layout models-workspace-page">
      <header className="models-page-header">
        <div className="models-page-header-main">
          <div>
            <span className="eyebrow">MODEL REGISTRY / {dataset.task_type.toUpperCase()}</span>
            <h1>模型库</h1>
            <p className="muted">管理受管训练产物，支持快速测试、split 评估与 FP32 ONNX 导出。</p>
          </div>
        </div>
        <div className="models-page-header-meta">
          <span className="models-chip"><strong>{dataset.name}</strong></span>
          <span className="models-chip">{taskTypeLabel(dataset.task_type)}</span>
          <span className="models-chip models-chip--accent">{models.length} 个模型</span>
        </div>
      </header>

      {error && <div className="validation invalid"><strong>模型操作失败</strong><span>{error}</span></div>}
      {notice && <p className="training-workspace-notice">{notice}</p>}

      <ModelLibraryStats
        total={models.length}
        ptCount={filterCounts.pt}
        onnxCount={filterCounts.onnx}
        bestMap50={formatMetric(bestModel?.map50)}
      />

      {bestModel && (
        <ModelLibraryHero
          bestModel={bestModel}
          dataset={dataset}
          busy={busy}
          onTest={setTesting}
          onOpenDetail={(model) => setDetailId(model.id)}
          onExportOnnx={(model) => void exportOnnx(model)}
        />
      )}

      {comparison && (
        <section className="models-comparison-result">
          <div className="models-comparison-result-head">
            <div>
              <span className="eyebrow">MODEL COMPARISON</span>
              <h3>{comparison.candidate.name} vs {comparison.baseline.name}</h3>
            </div>
            <button type="button" className="button" onClick={() => setComparison(undefined)}>关闭</button>
          </div>
          <div className="models-comparison-deltas">
            {Object.entries(comparison.deltas).map(([key, value]) => (
              <span key={key}>
                <small>{key}</small>
                <strong className={typeof value === "number" ? (value > 0 ? "positive" : value < 0 ? "negative" : undefined) : undefined}>
                  {formatDelta(value)}
                </strong>
              </span>
            ))}
          </div>
          {comparison.suggestions[0] && <p className="muted">{comparison.suggestions[0]}</p>}
        </section>
      )}

      <section className="models-library-section">
        <div className="models-library-toolbar">
          <div>
            <h2>模型列表</h2>
            <p className="models-library-sub">共 {models.length} 个 · 当前显示 {filteredModels.length} 个</p>
          </div>
          <div className="models-library-toolbar-actions">
            <label className="models-library-search">
              <IconSearch size={14} />
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="搜索名称 / 版本 / 格式"
              />
            </label>
            <select className="models-sort-select" value={listSort} onChange={(event) => setListSort(event.target.value as ModelListSort)} aria-label="排序方式">
              <option value="newest">最新优先</option>
              <option value="map50">mAP50 优先</option>
            </select>
            <button
              type="button"
              className={`models-filter-toggle${includeArchived ? " active" : ""}`}
              onClick={() => setIncludeArchived((value) => !value)}
            >
              {includeArchived ? "含归档" : "仅启用"}
            </button>
          </div>
        </div>

        <div className="models-library-filters" role="tablist" aria-label="模型筛选">
          {([
            ["all", "全部"],
            ["pt", "PT"],
            ["onnx", "ONNX"],
            ["best", "Best"],
            ["archived", "归档"],
          ] as const).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={listFilter === id}
              className={`models-library-filter${listFilter === id ? " active" : ""}`}
              onClick={() => setListFilter(id)}
            >
              {label}
              <em>{filterCounts[id]}</em>
            </button>
          ))}
        </div>

        {!filteredModels.length ? (
          <div className="models-library-empty">
            <strong>{models.length ? "没有匹配的模型" : "暂无模型"}</strong>
            <p>{models.length ? "试试调整搜索、筛选或归档开关。" : "完成训练后，best.pt 与 last.pt 会自动进入模型库。"}</p>
            {!!models.length && (
              <button type="button" className="button" onClick={() => { setSearch(""); setListFilter("all"); }}>
                清除筛选
              </button>
            )}
          </div>
        ) : (
          <div className="library-grid">
            {filteredModels.map((model) => (
              <ModelLibraryCard
                key={model.id}
                model={model}
                models={models}
                datasetName={dataset.name}
                selected={selectedIds.includes(model.id)}
                busy={busy}
                onToggleSelect={(id) => setSelectedIds((ids) => ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id])}
                onTest={setTesting}
                onOpenDetail={(item) => setDetailId(item.id)}
                onExportOnnx={(item) => void exportOnnx(item)}
                onDelete={remove}
              />
            ))}
          </div>
        )}
      </section>

      <ModelComparisonBar
        selectedIds={selectedIds}
        models={models}
        compareBusy={compareBusy}
        busy={busy}
        onCompareSelected={() => void compareSelected()}
        onCompareWithBest={() => void compareWithBest()}
        onDeleteSelected={removeSelected}
        onDismiss={() => setSelectedIds([])}
      />

      {testing && <ModelTestModal model={testing} onClose={() => setTesting(undefined)} />}
    </main>
  );
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。";
}
