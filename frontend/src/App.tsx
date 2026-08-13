import { useEffect, useMemo, useRef, useState } from "react";
import { AnnotationCanvas, type AnnotationDraft } from "./annotation/AnnotationCanvas";
import { api, apiUrl } from "./api/client";
import { StarterShell, type StarterSection } from "./components/StarterShell";
import { TrainingView } from "./pages/TrainingView";
import { ModelsView } from "./pages/ModelsView";
import type { Annotation, BBox, ClassLabel, Dataset, DatasetQualityReport, DuplicateReport, ImageItem, SamCapabilities, SamPrediction, SplitName, TaskType, ValidationReport } from "./types";

type View = "workspace" | "dataset" | "annotation" | "training" | "models";

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset>();
  const [classes, setClasses] = useState<ClassLabel[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageItem>();
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [proposalDrafts, setProposalDrafts] = useState<AnnotationDraft[]>();
  const [view, setView] = useState<View>("workspace");
  const [duplicates, setDuplicates] = useState<DuplicateReport>();
  const [activeClassId, setActiveClassId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [samCapabilities, setSamCapabilities] = useState<SamCapabilities>();

  const refreshDatasets = async () => {
    const result = await api.listDatasets();
    setDatasets(result);
    if (selected) {
      const current = result.find((dataset) => dataset.id === selected.id);
      if (current) setSelected(current);
    }
  };

  const loadDataset = async (dataset: Dataset, nextView: View = "dataset") => {
    setSelected(dataset);
    setView(nextView);
    setSelectedImage(undefined);
    setProposalDrafts(undefined);
    setDuplicates(undefined);
    const [nextClasses, nextImages] = await Promise.all([api.listClasses(dataset.id), api.listImages(dataset.id)]);
    setClasses(nextClasses);
    setImages(nextImages.items);
    setActiveClassId((current) => current || nextClasses[0]?.id || "");
  };

  useEffect(() => {
    refreshDatasets().catch((reason: unknown) => setError(errorMessage(reason)));
    api.getSystemInfo().then((info) => setSamCapabilities(info.sam)).catch((reason: unknown) => setError(errorMessage(reason)));
  }, []);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    setNotice("");
    try { await action(); } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  };

  const runResult = async <T,>(action: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true);
    setError("");
    setNotice("");
    try { return await action(); } catch (reason) { setError(errorMessage(reason)); return undefined; } finally { setBusy(false); }
  };

  const openImage = async (image: ImageItem) => {
    if (!selected) return;
    await run(async () => {
      setSelectedImage(image);
      setAnnotations(await api.getAnnotations(selected.id, image.id));
      setProposalDrafts(undefined);
      setView("annotation");
    });
  };

  const saveAnnotations = async (drafts: AnnotationDraft[]) => {
    if (!selected || !selectedImage) return;
    await run(async () => {
      const saved = await api.replaceAnnotations(selected.id, selectedImage.id, drafts.map(({ id: _id, ...item }) => item));
      setAnnotations(saved);
      setImages((items) => items.map((item) => item.id === selectedImage.id ? { ...item, status: saved.length ? "annotated" : "unannotated" } : item));
      setNotice("标注已保存");
    });
  };

  const continueAnnotation = async (dataset: Dataset) => {
    await run(async () => {
      await loadDataset(dataset, "dataset");
      const nextImages = await api.listImages(dataset.id);
      const image = nextImages.items.find((item) => item.status === "unannotated") ?? nextImages.items[0];
      if (!image) {
        setNotice("该数据集还没有图片可标注。");
        return;
      }
      setSelectedImage(image);
      setAnnotations(await api.getAnnotations(dataset.id, image.id));
      setProposalDrafts(undefined);
      setView("annotation");
    });
  };

  const displayedDataset = selected;
  const activeSection: StarterSection = view === "annotation" || view === "dataset" ? "workspace" : view;
  const navigate = (section: StarterSection) => {
    if (section === "training" && !selected && datasets[0]) {
      void loadDataset(datasets[0], "training");
      return;
    }
    if (section !== "workspace" && !selected) {
      setNotice("请先创建或选择一个数据集。");
      return;
    }
    setView(section);
  };

  return (
    <StarterShell active={activeSection} datasetName={selected?.name} onNavigate={navigate}>
      {error && <div className="toast error">{error}</div>}
      {notice && <div className="toast success">{notice}</div>}
      {view === "annotation" && displayedDataset && selectedImage ? (
        <AnnotationView
          dataset={displayedDataset}
          image={selectedImage}
          classes={classes}
          annotations={annotations}
          initialDrafts={proposalDrafts}
          activeClassId={activeClassId}
          onClassChange={setActiveClassId}
          onBack={() => setView("dataset")}
          onPrevious={() => { const index = images.findIndex((item) => item.id === selectedImage.id); if (index > 0) void openImage(images[index - 1]); }}
          onNext={() => { const index = images.findIndex((item) => item.id === selectedImage.id); if (index >= 0 && index < images.length - 1) void openImage(images[index + 1]); }}
          hasPrevious={images.findIndex((item) => item.id === selectedImage.id) > 0}
          hasNext={images.findIndex((item) => item.id === selectedImage.id) < images.length - 1}
          onSave={saveAnnotations}
          onSam={async (box) => {
            setBusy(true);
            setError("");
            try { return await api.samPredict({ image_id: selectedImage.id, class_id: activeClassId, prompt_type: "box", box }); }
            catch (reason) { setError(errorMessage(reason)); throw reason; }
            finally { setBusy(false); }
          }}
          onSamPoints={async (points) => {
            setBusy(true);
            setError("");
            try { return await api.samPredict({ image_id: selectedImage.id, class_id: activeClassId, prompt_type: "point", points: points.map(([x, y]) => ({ x, y, label: 1 })) }); }
            catch (reason) { setError(errorMessage(reason)); throw reason; }
            finally { setBusy(false); }
          }}
          busy={busy}
          samCapabilities={samCapabilities}
        />
      ) : view === "training" && displayedDataset ? (
        <TrainingView datasets={datasets} dataset={displayedDataset} onDatasetChange={(dataset) => void loadDataset(dataset, "training")} onBack={() => setView("workspace")} />
      ) : view === "models" && displayedDataset ? (
        <ModelsView dataset={displayedDataset} onBack={() => setView("workspace")} onOpenPreannotated={(image, drafts) => { setSelectedImage(image); setAnnotations([]); setProposalDrafts(drafts as AnnotationDraft[]); setView("annotation"); }} />
      ) : view === "dataset" && displayedDataset ? (
        <DatasetDetail
          selected={displayedDataset}
          classes={classes}
          images={images}
          duplicates={duplicates}
          activeClassId={activeClassId}
          busy={busy}
          onBack={() => setView("workspace")}
          onAddClass={(name) => selected && run(async () => { const item = await api.createClass(selected.id, name); setClasses((items) => [...items, item]); setActiveClassId(item.id); setNotice("类别已添加"); })}
          onUpload={(files, split) => selected && run(async () => { const result = await api.uploadImages(selected.id, files, split); setImages((items) => [...items, ...result.items]); await refreshDatasets(); setNotice(`已导入 ${result.imported} 张图片`); })}
          onScan={(path, split) => selected && run(async () => { const result = await api.scanImages(selected.id, path, split); const nextImages = await api.listImages(selected.id); setImages(nextImages.items); await refreshDatasets(); setNotice(`扫描完成：导入 ${result.imported} 张，跳过 ${result.skipped} 张`); })}
          onDuplicates={() => selected && run(async () => setDuplicates(await api.duplicateReport(selected.id)))}
          onVideo={(file, split, interval) => selected && run(async () => { const result = await api.importVideo(selected.id, file, split, interval); const next = await api.listImages(selected.id); setImages(next.items); setNotice(`已从视频导入 ${result.imported} 帧`); await refreshDatasets(); })}
          onActiveClassChange={setActiveClassId}
          onOpenImage={openImage}
          onOpenTraining={() => navigate("training")}
          onOpenModels={() => navigate("models")}
          onSplitChange={(image, split) => selected && run(async () => { const updated = await api.updateImageSplit(selected.id, image.id, split); setImages((items) => items.map((item) => item.id === image.id ? updated : item)); await refreshDatasets(); })}
          onBulkSplit={(ids, split) => selected && run(async () => { const result = await api.updateImageSplits(selected.id, ids, split); const next = await api.listImages(selected.id); setImages(next.items); setNotice(`已更新 ${result.updated} 张图片的 split`); await refreshDatasets(); })}
          onAutoSplit={() => selected && run(async () => { const result = await api.autoSplitImages(selected.id, { train_ratio: .8, val_ratio: .1, test_ratio: .1, seed: 42 }); const next = await api.listImages(selected.id); setImages(next.items); setNotice(`已按 80/10/10 分配 ${result.updated} 张图片`); await refreshDatasets(); })}
          onTile={(name, tileSize, overlap, keepEmpty) => selected && run(async () => { const result = await api.tileDataset(selected.id, { name, tile_size: tileSize, overlap, keep_empty_tiles: keepEmpty }); await refreshDatasets(); const datasets = await api.listDatasets(); const derived = datasets.find((item) => item.id === result.dataset_id); if (derived) await loadDataset(derived); setNotice(`已生成切片数据集：${result.generated_images} 张图片 / ${result.generated_annotations} 个标注`); })}
        />
      ) : (
        <DatasetHome
          datasets={datasets}
          busy={busy}
          onSelect={(dataset) => run(() => loadDataset(dataset))}
          onCreate={(name, taskType) => run(async () => { const dataset = await api.createDataset(name, taskType); await refreshDatasets(); await loadDataset(dataset); setNotice("数据集已创建"); })}
          onImport={(file, name, taskType, format) => run(async () => { const result = format === "coco" ? await api.importCoco(file, name, taskType) : await api.importYolo(file, name, taskType); await refreshDatasets(); await loadDataset(result.dataset); setNotice(`已导入 ${result.imported_images} 张图片和 ${result.imported_annotations} 个标注`); })}
          onValidate={(dataset) => runResult(() => api.validateDataset(dataset.id))}
          onQuality={(dataset) => runResult(() => api.qualityReport(dataset.id))}
          onContinueAnnotation={(dataset) => void continueAnnotation(dataset)}
          onDelete={(dataset) => run(async () => {
            await api.deleteDataset(dataset.id);
            setDatasets((items) => items.filter((item) => item.id !== dataset.id));
            if (selected?.id === dataset.id) {
              setSelected(undefined);
              setClasses([]);
              setImages([]);
              setSelectedImage(undefined);
              setAnnotations([]);
              setProposalDrafts(undefined);
              setDuplicates(undefined);
              setActiveClassId("");
            }
            setNotice(`数据集“${dataset.name}”已删除`);
          })}
        />
      )}
    </StarterShell>
  );
}

export function DatasetHome(props: {
  datasets: Dataset[];
  busy: boolean;
  onSelect: (dataset: Dataset) => void;
  onCreate: (name: string, type: TaskType) => void;
  onImport: (file: File, name: string, type: TaskType, format: "yolo" | "coco") => void;
  onValidate: (dataset: Dataset) => Promise<ValidationReport | undefined>;
  onQuality: (dataset: Dataset) => Promise<DatasetQualityReport | undefined>;
  onContinueAnnotation: (dataset: Dataset) => void;
  onDelete: (dataset: Dataset) => void;
}) {
  const [dialog, setDialog] = useState<"create" | "import" | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Dataset>();
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [archive, setArchive] = useState<File>();
  const [format, setFormat] = useState<"yolo" | "coco">("yolo");
  const [validationReports, setValidationReports] = useState<Record<string, ValidationReport>>({});
  const [qualityReports, setQualityReports] = useState<Record<string, DatasetQualityReport>>({});
  const fileRef = useRef<HTMLInputElement>(null);

  const closeDialog = () => {
    setDialog(null);
    setName("");
    setTaskType("detect");
    setArchive(undefined);
    setFormat("yolo");
  };
  const create = () => {
    if (!name.trim()) return;
    props.onCreate(name.trim(), taskType);
    closeDialog();
  };
  const importArchive = () => {
    if (!archive || !name.trim()) return;
    props.onImport(archive, name.trim(), taskType, format);
    closeDialog();
  };

  return <main className="dataset-home-page">
    <header className="dataset-home-header">
      <div><h1>数据集</h1><p>在这里创建、导入和管理你的 YOLO 数据集。</p></div>
      <div className="header-actions"><button className="button" onClick={() => setDialog("import")}>导入数据</button><button className="button primary" onClick={() => setDialog("create")}>新建数据集</button></div>
    </header>
    <section className="dataset-overview">
      <div className="section-title-row"><div><span className="eyebrow">ALL DATASETS</span><h2>所有数据集</h2></div><span className="dataset-total">{props.datasets.length} 个数据集</span></div>
      <div className="dataset-grid">
        {props.datasets.map((dataset, index) => <article key={dataset.id} className="dataset-card">
          <header><span className={`dataset-icon tone-${index % 3}`}>{dataset.task_type === "segment" || dataset.task_type === "obb" ? "◇" : dataset.task_type === "classify" ? "○" : "□"}</span><div><h2>{dataset.name}</h2><small>{taskDescription(dataset.task_type)}</small></div><span className="status ready">就绪</span></header>
          <div className="dataset-metrics"><span><small>图片</small><strong>{dataset.image_count.toLocaleString()}</strong></span><span><small>类别</small><strong>{dataset.class_count.toLocaleString()}</strong></span></div>
          <div className="dataset-progress"><span>标注进度</span><strong>{dataset.image_count ? `${Math.round(dataset.annotated_image_count / dataset.image_count * 100)}%` : "等待导入"}</strong></div>
          <div className="progress-track"><i style={{ width: `${dataset.image_count ? dataset.annotated_image_count / dataset.image_count * 100 : 0}%` }} /></div>
          <button className="button primary dataset-primary-action" onClick={() => props.onSelect(dataset)}>打开数据集</button>
          <div className="dataset-card-actions"><button className="button" disabled={props.busy || !dataset.image_count} onClick={() => props.onContinueAnnotation(dataset)}>继续标注</button><a className="button" href={api.exportYoloUrl(dataset.id)}>导出 YOLO</a>{["detect", "segment"].includes(dataset.task_type) ? <a className="button" href={api.exportCocoUrl(dataset.id)}>导出 COCO</a> : <span aria-hidden="true" />}<button className="button" disabled={props.busy} onClick={() => void props.onValidate(dataset).then((report) => report && setValidationReports((items) => ({ ...items, [dataset.id]: report })))}>运行校验</button><button className="button" disabled={props.busy} onClick={() => void props.onQuality(dataset).then((report) => report && setQualityReports((items) => ({ ...items, [dataset.id]: report })))}>质量报告</button><button className="button dataset-delete-action" disabled={props.busy} onClick={() => setDeleteTarget(dataset)}>删除</button></div>
          {validationReports[dataset.id] && <ValidationPanel report={validationReports[dataset.id]} compact />}
          {qualityReports[dataset.id] && <QualityPanel report={qualityReports[dataset.id]} compact />}
        </article>)}
        {!props.datasets.length && <section className="dataset-empty-card"><span className="dataset-empty-icon">□</span><h2>还没有数据集</h2><p>新建一个空数据集，或导入已有的 YOLO ZIP 数据集。</p><div><button className="button" onClick={() => setDialog("import")}>导入数据</button><button className="button primary" onClick={() => setDialog("create")}>新建数据集</button></div></section>}
      </div>
    </section>
    {dialog && <div className="modal-backdrop" role="presentation" onMouseDown={closeDialog}>
      <section className="dataset-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">{dialog === "create" ? "NEW DATASET" : "IMPORT DATASET"}</span><h2 id="dataset-dialog-title">{dialog === "create" ? "新建数据集" : `导入 ${format === "yolo" ? "YOLO" : "COCO"} 数据集`}</h2><p>{dialog === "create" ? "创建后即可上传图片并开始标注。" : format === "yolo" ? "导入包含 data.yaml、图片和标签的 YOLO ZIP 文件。" : "导入包含 annotations.json（或 instances.json）和图片的 COCO ZIP 文件。"}</p></div><button className="icon-button" onClick={closeDialog} aria-label="关闭">×</button></header>
        {dialog === "import" && <div className="import-format"><label>导入格式<select value={format} onChange={(event) => setFormat(event.target.value as "yolo" | "coco")}><option value="yolo">YOLO ZIP</option><option value="coco">COCO ZIP（detect / segment）</option></select></label><span>{format === "yolo" ? "ZIP 内应包含 data.yaml、images 和 labels" : "ZIP 内应包含 annotations.json/instances.json 与 images"}</span></div>}
        <label>数据集名称<input value={name} placeholder={dialog === "import" ? "例如：road-signs" : "例如：my-dataset"} onChange={(event) => setName(event.target.value)} autoFocus /></label>
        <label>任务类型<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}><option value="detect">目标检测（Bounding Box）</option><option value="segment">实例分割（Polygon / SAM）</option><option value="obb">旋转框（OBB）</option><option value="classify">图像分类</option></select></label>
        {dialog === "import" && <><input ref={fileRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => setArchive(event.target.files?.[0])} /><button className="import-file-picker" onClick={() => fileRef.current?.click()}><span>⇧</span><strong>{archive?.name ?? `选择 ${format.toUpperCase()} ZIP 文件`}</strong><small>{archive ? `${Math.ceil(archive.size / 1024)} KB` : format === "yolo" ? "ZIP 内应包含 data.yaml、images 和 labels" : "ZIP 内应包含 annotations.json/instances.json 与 images"}</small></button></>}
        <footer><button className="button" onClick={closeDialog}>取消</button><button className="button primary" disabled={props.busy || !name.trim() || (dialog === "import" && !archive)} onClick={dialog === "create" ? create : importArchive}>{dialog === "create" ? "创建数据集" : "开始导入"}</button></footer>
      </section>
    </div>}
    {deleteTarget && <div className="modal-backdrop" role="presentation" onMouseDown={() => setDeleteTarget(undefined)}>
      <section className="dataset-dialog dataset-delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="dataset-delete-dialog-title" aria-describedby="dataset-delete-dialog-description" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">DELETE DATASET</span><h2 id="dataset-delete-dialog-title">删除数据集“{deleteTarget.name}”？</h2><p id="dataset-delete-dialog-description">这会永久删除该数据集的 {deleteTarget.image_count} 张图片、类别和标注，且无法撤销。</p></div><button className="icon-button" onClick={() => setDeleteTarget(undefined)} aria-label="关闭">×</button></header>
        <footer><button className="button" disabled={props.busy} onClick={() => setDeleteTarget(undefined)}>取消</button><button className="button danger" disabled={props.busy} onClick={() => { props.onDelete(deleteTarget); setDeleteTarget(undefined); }}>确认删除</button></footer>
      </section>
    </div>}
  </main>;
}

function DatasetDetail(props: {
  selected: Dataset; classes: ClassLabel[]; images: ImageItem[]; duplicates?: DuplicateReport; activeClassId: string; busy: boolean;
  onBack: () => void; onAddClass: (name: string) => void;
  onUpload: (files: File[], split: SplitName) => void; onScan: (path: string, split: SplitName) => void; onDuplicates: () => void; onVideo: (file: File, split: SplitName, interval: number) => void;
  onActiveClassChange: (id: string) => void; onOpenImage: (image: ImageItem) => void; onOpenTraining: () => void; onOpenModels: () => void; onSplitChange: (image: ImageItem, split: SplitName) => void; onBulkSplit: (ids: string[], split: SplitName) => void; onAutoSplit: () => void; onTile: (name: string, tileSize: number, overlap: number, keepEmpty: boolean) => void;
}) {
  const [className, setClassName] = useState("");
  const [split, setSplit] = useState<SplitName>("train");
  const [scanPath, setScanPath] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<SplitName | "all">("all");
  const [status, setStatus] = useState<"all" | "annotated" | "unannotated">("all");
  const [checked, setChecked] = useState<string[]>([]);
  const [tileSize, setTileSize] = useState(1024);
  const [tileOverlap, setTileOverlap] = useState(0.2);
  const [keepEmptyTiles, setKeepEmptyTiles] = useState(false);
  const visibleImages = useMemo(() => props.images.filter((image) => (tab === "all" || image.split === tab) && (status === "all" || image.status === status)), [props.images, tab, status]);
  const toggle = (id: string) => setChecked((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]);

  return (
    <main className="dataset-detail-page">
      <div className="dataset-breadcrumbs"><button onClick={props.onBack}>数据集</button><span>/</span><strong>{props.selected.name}</strong></div>
      <section className="workspace panel">
          <div className="workspace-header"><div><span className="eyebrow">DATASET WORKSPACE</span><h1>{props.selected.name}</h1><p>{taskDescription(props.selected.task_type)} · {props.selected.image_count} 张图片 · {props.selected.class_count} 个类别</p></div><div className="header-actions"><button className="button primary" onClick={props.onOpenTraining}>训练</button><button className="button" onClick={props.onOpenModels}>模型</button><a className="button" href={api.exportYoloUrl(props.selected.id)}>导出 YOLO ZIP</a>{["detect", "segment"].includes(props.selected.task_type) && <a className="button" href={api.exportCocoUrl(props.selected.id)}>导出 COCO</a>}</div></div>
          <div className="action-row"><input ref={uploadRef} type="file" accept="image/*" multiple hidden onChange={(event) => { if (event.target.files) props.onUpload(Array.from(event.target.files), split); event.currentTarget.value = ""; }} /><input ref={videoRef} type="file" accept="video/mp4,video/quicktime,video/x-msvideo" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) props.onVideo(file, split, 30); event.currentTarget.value = ""; }} /><select value={split} onChange={(event) => setSplit(event.target.value as SplitName)}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select><button className="button primary" onClick={() => uploadRef.current?.click()}>上传图片</button><button className="button" onClick={() => videoRef.current?.click()}>视频抽帧（每30帧）</button><input placeholder="扫描受管目录相对路径" value={scanPath} onChange={(event) => setScanPath(event.target.value)} /><button className="button" disabled={!scanPath.trim()} onClick={() => props.onScan(scanPath.trim(), split)}>扫描目录</button><button className="button" onClick={props.onDuplicates}>重复/相似图</button><button className="button" disabled={props.busy} onClick={props.onAutoSplit}>自动划分 80/10/10</button></div>
          {(["detect", "segment"] as string[]).includes(props.selected.task_type) && <div className="bulk-bar"><strong>生成切片数据集</strong><select value={tileSize} onChange={(event) => setTileSize(Number(event.target.value))}><option value={512}>512 px</option><option value={1024}>1024 px</option><option value={1280}>1280 px</option></select><select value={tileOverlap} onChange={(event) => setTileOverlap(Number(event.target.value))}><option value={0}>无重叠</option><option value={0.1}>10% overlap</option><option value={0.2}>20% overlap</option></select><label><input type="checkbox" checked={keepEmptyTiles} onChange={(event) => setKeepEmptyTiles(event.target.checked)} /> 保留空切片</label><button className="button" disabled={props.busy || !props.images.length} onClick={() => props.onTile(`${props.selected.name}-tiles-${tileSize}`, tileSize, tileOverlap, keepEmptyTiles)}>生成新数据集</button></div>}
          <div className="split-tabs">{(["all", "train", "val", "test"] as const).map((item) => <button key={item} className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>{item === "all" ? "全部" : item}</button>)}<select className="image-filter" value={status} onChange={(event) => setStatus(event.target.value as typeof status)}><option value="all">全部状态</option><option value="unannotated">未标注</option><option value="annotated">已标注</option></select></div>
          <div className="class-strip"><span className="eyebrow">CLASSES</span>{props.classes.map((item) => <button key={item.id} className={props.activeClassId === item.id ? "class-chip active" : "class-chip"} onClick={() => props.onActiveClassChange(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}<input value={className} placeholder="新增类别" onChange={(event) => setClassName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && className.trim()) { props.onAddClass(className.trim()); setClassName(""); } }} /></div>
          {props.duplicates && <DuplicatePanel report={props.duplicates} />}
          {checked.length > 0 && <div className="bulk-bar"><strong>已选择 {checked.length} 张</strong><button className="button" onClick={() => props.onBulkSplit(checked, "train")}>设为 train</button><button className="button" onClick={() => props.onBulkSplit(checked, "val")}>设为 val</button><button className="button" onClick={() => props.onBulkSplit(checked, "test")}>设为 test</button><button className="button" onClick={() => setChecked([])}>取消选择</button></div>}
          <div className="image-grid">{visibleImages.map((image) => <article className="image-card" key={image.id}><label className="image-check"><input type="checkbox" checked={checked.includes(image.id)} onChange={() => toggle(image.id)} />选择</label><button onClick={() => props.onOpenImage(image)}><img src={apiUrl(image.file_url)} alt={image.file_name} /><div><strong>{image.file_name}</strong><span>{image.status === "annotated" ? "已标注" : "未标注"}</span><select value={image.split} onClick={(event) => event.stopPropagation()} onChange={(event) => { event.stopPropagation(); props.onSplitChange(image, event.target.value as SplitName); }}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select></div></button></article>)}{!visibleImages.length && <div className="empty-state"><strong>开始导入图片</strong><span>上传图片后，在这里选择一张进入标注工作台。</span></div>}</div>
      </section>
    </main>
  );
}

function QualityPanel({ report, compact = false }: { report: DatasetQualityReport; compact?: boolean }) { return <section className={compact ? "quality-panel compact-quality-panel" : "quality-panel"}><div className="quality-summary"><span><small>标注覆盖率</small><strong>{Math.round(report.summary.coverage * 100)}%</strong></span><span><small>标注总数</small><strong>{report.summary.annotation_count}</strong></span><span><small>小目标</small><strong>{report.summary.small_object_count}</strong></span><span><small>问题</small><strong>{report.issues.length}</strong></span></div>{!compact && <div className="quality-grid"><div><h3>类别分布</h3>{report.class_distribution.map((item) => <p key={item.class_id}><span>{item.class_index}. {item.name}</span><b>{item.count}</b><i><em style={{ width: `${Math.max(item.ratio * 100, item.count ? 3 : 0)}%` }} /></i></p>)}</div><div><h3>需要关注</h3>{report.issues.slice(0, 5).map((item, index) => <p key={`${item.type}-${index}`}><b>{item.level}</b><span>{item.message}</span></p>)}{!report.issues.length && <p>当前未发现需要关注的问题。</p>}</div></div>}</section>; }
function DuplicatePanel({ report }: { report: DuplicateReport }) { return <section className="quality-panel"><div className="quality-summary"><span><small>精确重复</small><strong>{report.duplicate}</strong></span><span><small>相似图片</small><strong>{report.similar}</strong></span><span><small>无效图片</small><strong>{report.invalid_images}</strong></span><span><small>分析图片</small><strong>{report.images}</strong></span></div><div className="quality-grid"><div><h3>相似性分组</h3>{report.groups.slice(0, 6).map((item, index) => <p key={index}><span>{item.kind === "exact" ? "精确重复" : "相似"} · {item.image_ids.length} 张</span><b>{(item.score * 100).toFixed(0)}%</b></p>)}{!report.groups.length && <p>未发现重复或相似图片。</p>}</div><div><h3>处理建议</h3><p>重复检测为只读报告，不会删除任何图片。</p><p>相似图应由你在标注和 split 前人工决定是否保留。</p></div></div></section>; }


function ValidationPanel({ report, compact = false }: { report: ValidationReport; compact?: boolean }) {
  return <div className={`${report.valid ? "validation valid" : "validation invalid"}${compact ? " compact-validation" : ""}`}><strong>{report.valid ? "校验通过" : "发现需要处理的问题"}</strong><span>{report.error_count} errors · {report.warning_count} warnings</span>{!compact && report.issues.slice(0, 3).map((issue) => <small key={`${issue.code}-${issue.image_id ?? "dataset"}`}>{issue.level === "error" ? "!" : "·"} {issue.message}</small>)}</div>;
}

export function AnnotationView(props: { dataset: Dataset; image: ImageItem; classes: ClassLabel[]; annotations: Annotation[]; initialDrafts?: AnnotationDraft[]; activeClassId: string; onClassChange: (id: string) => void; onBack: () => void; onPrevious?: () => void; onNext?: () => void; hasPrevious?: boolean; hasNext?: boolean; onSave: (drafts: AnnotationDraft[]) => void; onSam: (box: BBox) => Promise<SamPrediction>; onSamPoints: (points: [number, number][]) => Promise<SamPrediction>; busy: boolean; samCapabilities?: SamCapabilities }) {
  const toDraft = (item: Annotation): AnnotationDraft => ({ id: item.id, class_id: item.class_id, type: item.type, bbox: item.bbox ?? undefined, polygon: item.polygon ?? undefined, obb: item.obb ?? undefined, source: item.source });
  const [drafts, setDrafts] = useState<AnnotationDraft[]>(props.initialDrafts ?? props.annotations.map(toDraft));
  useEffect(() => setDrafts(props.initialDrafts ?? props.annotations.map(toDraft)), [props.annotations, props.initialDrafts, props.image.id]);
  const requestSam = async (box: BBox) => {
    try {
      const suggestion = await props.onSam(box);
      setDrafts((items) => [...items, { class_id: suggestion.class_id, type: "polygon", polygon: suggestion.polygon, source: suggestion.backend_used === "ultralytics_sam" ? "sam" : "manual" }]);
    } catch {
      // The page-level request handler has already surfaced the error.
    }
  };
  const requestSamPoints = async (points: [number, number][]) => {
    try {
      const suggestion = await props.onSamPoints(points);
      setDrafts((items) => [...items, { class_id: suggestion.class_id, type: "polygon", polygon: suggestion.polygon, source: suggestion.backend_used === "ultralytics_sam" ? "sam" : "manual" }]);
    } catch {
      // The page-level request handler has already surfaced the error.
    }
  };
  const annotated = drafts.map((draft, index) => ({ ...draft, id: draft.id ?? `draft-${index}`, image_id: props.image.id, dataset_id: props.dataset.id, class_index: 0, label: "", color: props.classes.find((item) => item.id === draft.class_id)?.color ?? "#f97316", created_at: "", updated_at: "" })) as Annotation[];
  const chooseClass = (classId: string) => setDrafts([{ class_id: classId, type: "classify", source: "manual" }]);
  useEffect(() => { const onKey = (event: KeyboardEvent) => { if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return; if (event.key === "ArrowLeft" && props.hasPrevious) props.onPrevious?.(); if (event.key === "ArrowRight" && props.hasNext) props.onNext?.(); if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") { event.preventDefault(); props.onSave(drafts); } }; window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, [drafts, props]);
  return <main className="annotation-layout"><div className="annotation-head"><button className="button" onClick={props.onBack}>← 返回数据集</button><div><span className="eyebrow">{props.dataset.name} / {props.dataset.task_type}</span><h1>{props.image.file_name}</h1><p className="muted">←/→ 切换图片 · ⌘/Ctrl+S 保存</p></div><div className="annotation-actions">{props.onPrevious && <button className="button" disabled={!props.hasPrevious} onClick={props.onPrevious}>← 上一张</button>}{props.onNext && <button className="button" disabled={!props.hasNext} onClick={props.onNext}>下一张 →</button>}<select value={props.activeClassId} onChange={(event) => props.onClassChange(event.target.value)}>{props.classes.map((item) => <option key={item.id} value={item.id}>{item.class_index}: {item.name}</option>)}</select><button className="button primary" disabled={props.busy || !props.activeClassId} onClick={() => props.onSave(drafts)}>保存标注</button></div></div><div className="annotation-body">{props.dataset.task_type === "classify" ? <section className="canvas-card classification-card"><span className="eyebrow">IMAGE CLASSIFICATION</span><h2>选择这张图片的唯一类别</h2><p className="hint">分类数据集每张图片只能保存一个类别，训练时会按 train/类别名/图片 导出。</p><div className="class-strip">{props.classes.map((item) => <button key={item.id} className={drafts[0]?.class_id === item.id ? "class-chip active" : "class-chip"} onClick={() => chooseClass(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}</div></section> : <AnnotationCanvas image={props.image} classes={props.classes} annotations={annotated} activeClassId={props.activeClassId} taskType={props.dataset.task_type} onChange={setDrafts} onSamBox={props.dataset.task_type === "segment" ? requestSam : undefined} onSamPoints={props.dataset.task_type === "segment" ? requestSamPoints : undefined} samBusy={props.busy} samCapabilities={props.samCapabilities} />}<aside className="annotation-sidebar panel"><span className="eyebrow">SAVED SHAPES</span><h2>{drafts.length} 个标注</h2>{drafts.map((draft, index) => <div className="shape-row" key={draft.id ?? index}><span className="shape-index">{index + 1}</span><div><strong>{props.classes.find((item) => item.id === draft.class_id)?.name ?? "Unknown"}</strong><small>{annotationLabel(draft.type)}</small></div><button className="icon-button" onClick={() => setDrafts((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}{!drafts.length && <p className="muted">{props.dataset.task_type === "classify" ? "请选择一个类别。" : "在图片上拖拽或点击创建标注。"}</p>}</aside></div></main>;
}

function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
function annotationLabel(type: Annotation["type"]): string { return ({ bbox: "Bounding box", polygon: "Polygon", obb: "Oriented bounding box", classify: "Image classification" })[type]; }
function taskDescription(taskType: TaskType): string { return ({ detect: "目标检测 · Bounding Box", segment: "实例分割 · Polygon / SAM", obb: "旋转框检测 · OBB", classify: "图像分类" })[taskType]; }
