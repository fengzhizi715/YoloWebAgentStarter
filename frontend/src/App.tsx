import { useEffect, useMemo, useRef, useState } from "react";
import { AnnotationCanvas, type AnnotationDraft } from "./annotation/AnnotationCanvas";
import { api, apiUrl } from "./api/client";
import { StarterShell, type StarterSection } from "./components/StarterShell";
import { TrainingView } from "./pages/TrainingView";
import { ModelsView } from "./pages/ModelsView";
import type { Annotation, BBox, ClassLabel, Dataset, ImageItem, SamCapabilities, SamPrediction, SplitName, TaskType, ValidationReport } from "./types";

type View = "workspace" | "dataset" | "annotation" | "training" | "models";

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset>();
  const [classes, setClasses] = useState<ClassLabel[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageItem>();
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [view, setView] = useState<View>("workspace");
  const [report, setReport] = useState<ValidationReport>();
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
    setReport(undefined);
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

  const openImage = async (image: ImageItem) => {
    if (!selected) return;
    await run(async () => {
      setSelectedImage(image);
      setAnnotations(await api.getAnnotations(selected.id, image.id));
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

  const displayedDataset = selected;
  const activeSection: StarterSection = view === "annotation" || view === "dataset" ? "workspace" : view;
  const navigate = (section: StarterSection) => {
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
          activeClassId={activeClassId}
          onClassChange={setActiveClassId}
          onBack={() => setView("workspace")}
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
        <TrainingView dataset={displayedDataset} onBack={() => setView("workspace")} />
      ) : view === "models" && displayedDataset ? (
        <ModelsView dataset={displayedDataset} onBack={() => setView("workspace")} />
      ) : view === "dataset" && displayedDataset ? (
        <DatasetDetail
          selected={displayedDataset}
          classes={classes}
          images={images}
          report={report}
          activeClassId={activeClassId}
          busy={busy}
          onBack={() => setView("workspace")}
          onAddClass={(name) => selected && run(async () => { const item = await api.createClass(selected.id, name); setClasses((items) => [...items, item]); setActiveClassId(item.id); setNotice("类别已添加"); })}
          onUpload={(files, split) => selected && run(async () => { const result = await api.uploadImages(selected.id, files, split); setImages((items) => [...items, ...result.items]); await refreshDatasets(); setNotice(`已导入 ${result.imported} 张图片`); })}
          onScan={(path, split) => selected && run(async () => { const result = await api.scanImages(selected.id, path, split); const nextImages = await api.listImages(selected.id); setImages(nextImages.items); await refreshDatasets(); setNotice(`扫描完成：导入 ${result.imported} 张，跳过 ${result.skipped} 张`); })}
          onValidate={() => selected && run(async () => setReport(await api.validateDataset(selected.id)))}
          onActiveClassChange={setActiveClassId}
          onOpenImage={openImage}
          onOpenTraining={() => navigate("training")}
          onOpenModels={() => navigate("models")}
          onSplitChange={(image, split) => selected && run(async () => { const updated = await api.updateImageSplit(selected.id, image.id, split); setImages((items) => items.map((item) => item.id === image.id ? updated : item)); await refreshDatasets(); })}
        />
      ) : (
        <DatasetHome
          datasets={datasets}
          busy={busy}
          onSelect={(dataset) => run(() => loadDataset(dataset))}
          onCreate={(name, taskType) => run(async () => { const dataset = await api.createDataset(name, taskType); await refreshDatasets(); await loadDataset(dataset); setNotice("数据集已创建"); })}
          onImport={(file, name, taskType) => run(async () => { const result = await api.importYolo(file, name, taskType); await refreshDatasets(); await loadDataset(result.dataset); setNotice(`已导入 ${result.imported_images} 张图片和 ${result.imported_annotations} 个标注`); })}
        />
      )}
    </StarterShell>
  );
}

function DatasetHome(props: {
  datasets: Dataset[];
  busy: boolean;
  onSelect: (dataset: Dataset) => void;
  onCreate: (name: string, type: TaskType) => void;
  onImport: (file: File, name: string, type: TaskType) => void;
}) {
  const [dialog, setDialog] = useState<"create" | "import" | null>(null);
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [archive, setArchive] = useState<File>();
  const fileRef = useRef<HTMLInputElement>(null);

  const closeDialog = () => {
    setDialog(null);
    setName("");
    setTaskType("detect");
    setArchive(undefined);
  };
  const create = () => {
    if (!name.trim()) return;
    props.onCreate(name.trim(), taskType);
    closeDialog();
  };
  const importArchive = () => {
    if (!archive || !name.trim()) return;
    props.onImport(archive, name.trim(), taskType);
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
          <div className="dataset-progress"><span>标注进度</span><strong>{dataset.image_count ? "待标注" : "等待导入"}</strong></div>
          <div className="progress-track"><i /></div>
          <button className="button primary dataset-primary-action" onClick={() => props.onSelect(dataset)}>打开数据集</button>
          <div className="dataset-quick-actions"><button onClick={() => props.onSelect(dataset)}>图片与标注</button><button onClick={() => props.onSelect(dataset)}>数据集设置</button></div>
        </article>)}
        {!props.datasets.length && <section className="dataset-empty-card"><span className="dataset-empty-icon">□</span><h2>还没有数据集</h2><p>新建一个空数据集，或导入已有的 YOLO ZIP 数据集。</p><div><button className="button" onClick={() => setDialog("import")}>导入数据</button><button className="button primary" onClick={() => setDialog("create")}>新建数据集</button></div></section>}
      </div>
    </section>
    {dialog && <div className="modal-backdrop" role="presentation" onMouseDown={closeDialog}>
      <section className="dataset-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">{dialog === "create" ? "NEW DATASET" : "IMPORT DATASET"}</span><h2 id="dataset-dialog-title">{dialog === "create" ? "新建数据集" : "导入 YOLO 数据集"}</h2><p>{dialog === "create" ? "创建后即可上传图片并开始标注。" : "导入包含 data.yaml、图片和标签的 YOLO ZIP 文件。"}</p></div><button className="icon-button" onClick={closeDialog} aria-label="关闭">×</button></header>
        {dialog === "import" && <div className="import-format"><strong>YOLO ZIP</strong><span>社区版当前支持的导入格式</span></div>}
        <label>数据集名称<input value={name} placeholder={dialog === "import" ? "例如：road-signs" : "例如：my-dataset"} onChange={(event) => setName(event.target.value)} autoFocus /></label>
        <label>任务类型<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}><option value="detect">目标检测（Bounding Box）</option><option value="segment">实例分割（Polygon / SAM）</option><option value="obb">旋转框（OBB）</option><option value="classify">图像分类</option></select></label>
        {dialog === "import" && <><input ref={fileRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => setArchive(event.target.files?.[0])} /><button className="import-file-picker" onClick={() => fileRef.current?.click()}><span>⇧</span><strong>{archive?.name ?? "选择 YOLO ZIP 文件"}</strong><small>{archive ? `${Math.ceil(archive.size / 1024)} KB` : "ZIP 内应包含 data.yaml、images 和 labels"}</small></button></>}
        <footer><button className="button" onClick={closeDialog}>取消</button><button className="button primary" disabled={props.busy || !name.trim() || (dialog === "import" && !archive)} onClick={dialog === "create" ? create : importArchive}>{dialog === "create" ? "创建数据集" : "开始导入"}</button></footer>
      </section>
    </div>}
  </main>;
}

function DatasetDetail(props: {
  selected: Dataset; classes: ClassLabel[]; images: ImageItem[]; report?: ValidationReport; activeClassId: string; busy: boolean;
  onBack: () => void; onAddClass: (name: string) => void;
  onUpload: (files: File[], split: SplitName) => void; onScan: (path: string, split: SplitName) => void; onValidate: () => void;
  onActiveClassChange: (id: string) => void; onOpenImage: (image: ImageItem) => void; onOpenTraining: () => void; onOpenModels: () => void; onSplitChange: (image: ImageItem, split: SplitName) => void;
}) {
  const [className, setClassName] = useState("");
  const [split, setSplit] = useState<SplitName>("train");
  const [scanPath, setScanPath] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<SplitName | "all">("all");
  const visibleImages = useMemo(() => tab === "all" ? props.images : props.images.filter((image) => image.split === tab), [props.images, tab]);

  return (
    <main className="dataset-detail-page">
      <div className="dataset-breadcrumbs"><button onClick={props.onBack}>数据集</button><span>/</span><strong>{props.selected.name}</strong></div>
      <section className="workspace panel">
          <div className="workspace-header"><div><span className="eyebrow">DATASET WORKSPACE</span><h1>{props.selected.name}</h1><p>{taskDescription(props.selected.task_type)} · {props.selected.image_count} 张图片 · {props.selected.class_count} 个类别</p></div><div className="header-actions"><button className="button primary" onClick={props.onOpenTraining}>训练</button><button className="button" onClick={props.onOpenModels}>模型</button><a className="button" href={api.exportYoloUrl(props.selected.id)}>导出 YOLO ZIP</a></div></div>
          <div className="action-row"><input ref={uploadRef} type="file" accept="image/*" multiple hidden onChange={(event) => { if (event.target.files) props.onUpload(Array.from(event.target.files), split); event.currentTarget.value = ""; }} /><select value={split} onChange={(event) => setSplit(event.target.value as SplitName)}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select><button className="button primary" onClick={() => uploadRef.current?.click()}>上传图片</button><input placeholder="扫描受管目录相对路径" value={scanPath} onChange={(event) => setScanPath(event.target.value)} /><button className="button" disabled={!scanPath.trim()} onClick={() => props.onScan(scanPath.trim(), split)}>扫描目录</button><button className="button" onClick={props.onValidate}>运行校验</button></div>
          <div className="split-tabs">{(["all", "train", "val", "test"] as const).map((item) => <button key={item} className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>{item === "all" ? "全部" : item}</button>)}</div>
          <div className="class-strip"><span className="eyebrow">CLASSES</span>{props.classes.map((item) => <button key={item.id} className={props.activeClassId === item.id ? "class-chip active" : "class-chip"} onClick={() => props.onActiveClassChange(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}<input value={className} placeholder="新增类别" onChange={(event) => setClassName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && className.trim()) { props.onAddClass(className.trim()); setClassName(""); } }} /></div>
          {props.report && <ValidationPanel report={props.report} />}
          <div className="image-grid">{visibleImages.map((image) => <button className="image-card" key={image.id} onClick={() => props.onOpenImage(image)}><img src={apiUrl(image.file_url)} alt={image.file_name} /><div><strong>{image.file_name}</strong><span>{image.status === "annotated" ? "已标注" : "未标注"}</span><select value={image.split} onClick={(event) => event.stopPropagation()} onChange={(event) => { event.stopPropagation(); props.onSplitChange(image, event.target.value as SplitName); }}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select></div></button>)}{!visibleImages.length && <div className="empty-state"><strong>开始导入图片</strong><span>上传图片后，在这里选择一张进入标注工作台。</span></div>}</div>
      </section>
    </main>
  );
}


function ValidationPanel({ report }: { report: ValidationReport }) {
  return <div className={report.valid ? "validation valid" : "validation invalid"}><strong>{report.valid ? "校验通过" : "发现需要处理的问题"}</strong><span>{report.error_count} errors · {report.warning_count} warnings</span>{report.issues.slice(0, 3).map((issue) => <small key={`${issue.code}-${issue.image_id ?? "dataset"}`}>{issue.level === "error" ? "!" : "·"} {issue.message}</small>)}</div>;
}

function AnnotationView(props: { dataset: Dataset; image: ImageItem; classes: ClassLabel[]; annotations: Annotation[]; activeClassId: string; onClassChange: (id: string) => void; onBack: () => void; onSave: (drafts: AnnotationDraft[]) => void; onSam: (box: BBox) => Promise<SamPrediction>; onSamPoints: (points: [number, number][]) => Promise<SamPrediction>; busy: boolean; samCapabilities?: SamCapabilities }) {
  const toDraft = (item: Annotation): AnnotationDraft => ({ id: item.id, class_id: item.class_id, type: item.type, bbox: item.bbox ?? undefined, polygon: item.polygon ?? undefined, obb: item.obb ?? undefined, source: item.source });
  const [drafts, setDrafts] = useState<AnnotationDraft[]>(props.annotations.map(toDraft));
  useEffect(() => setDrafts(props.annotations.map(toDraft)), [props.annotations]);
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
  return <main className="annotation-layout"><div className="annotation-head"><button className="button" onClick={props.onBack}>← 返回数据集</button><div><span className="eyebrow">{props.dataset.name} / {props.dataset.task_type}</span><h1>{props.image.file_name}</h1></div><div className="annotation-actions"><select value={props.activeClassId} onChange={(event) => props.onClassChange(event.target.value)}>{props.classes.map((item) => <option key={item.id} value={item.id}>{item.class_index}: {item.name}</option>)}</select><button className="button primary" disabled={props.busy || !props.activeClassId} onClick={() => props.onSave(drafts)}>保存标注</button></div></div><div className="annotation-body">{props.dataset.task_type === "classify" ? <section className="canvas-card classification-card"><span className="eyebrow">IMAGE CLASSIFICATION</span><h2>选择这张图片的唯一类别</h2><p className="hint">分类数据集每张图片只能保存一个类别，训练时会按 train/类别名/图片 导出。</p><div className="class-strip">{props.classes.map((item) => <button key={item.id} className={drafts[0]?.class_id === item.id ? "class-chip active" : "class-chip"} onClick={() => chooseClass(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}</div></section> : <AnnotationCanvas image={props.image} classes={props.classes} annotations={annotated} activeClassId={props.activeClassId} taskType={props.dataset.task_type} onChange={setDrafts} onSamBox={props.dataset.task_type === "segment" ? requestSam : undefined} onSamPoints={props.dataset.task_type === "segment" ? requestSamPoints : undefined} samBusy={props.busy} samCapabilities={props.samCapabilities} />}<aside className="annotation-sidebar panel"><span className="eyebrow">SAVED SHAPES</span><h2>{drafts.length} 个标注</h2>{drafts.map((draft, index) => <div className="shape-row" key={draft.id ?? index}><span className="shape-index">{index + 1}</span><div><strong>{props.classes.find((item) => item.id === draft.class_id)?.name ?? "Unknown"}</strong><small>{annotationLabel(draft.type)}</small></div><button className="icon-button" onClick={() => setDrafts((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}{!drafts.length && <p className="muted">{props.dataset.task_type === "classify" ? "请选择一个类别。" : "在图片上拖拽或点击创建标注。"}</p>}</aside></div></main>;
}

function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
function annotationLabel(type: Annotation["type"]): string { return ({ bbox: "Bounding box", polygon: "Polygon", obb: "Oriented bounding box", classify: "Image classification" })[type]; }
function taskDescription(taskType: TaskType): string { return ({ detect: "目标检测 · Bounding Box", segment: "实例分割 · Polygon / SAM", obb: "旋转框检测 · OBB", classify: "图像分类" })[taskType]; }
