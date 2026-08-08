import { useEffect, useMemo, useRef, useState } from "react";
import { AnnotationCanvas, type AnnotationDraft } from "./annotation/AnnotationCanvas";
import { api, apiUrl } from "./api/client";
import { TrainingView } from "./pages/TrainingView";
import type { Annotation, ClassLabel, Dataset, ImageItem, SplitName, TaskType, ValidationReport } from "./types";

type View = "workspace" | "annotation" | "training";

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

  const refreshDatasets = async () => {
    const result = await api.listDatasets();
    setDatasets(result);
    if (selected) {
      const current = result.find((dataset) => dataset.id === selected.id);
      if (current) setSelected(current);
    }
  };

  const loadDataset = async (dataset: Dataset) => {
    setSelected(dataset);
    setView("workspace");
    setSelectedImage(undefined);
    setReport(undefined);
    const [nextClasses, nextImages] = await Promise.all([api.listClasses(dataset.id), api.listImages(dataset.id)]);
    setClasses(nextClasses);
    setImages(nextImages.items);
    setActiveClassId((current) => current || nextClasses[0]?.id || "");
  };

  useEffect(() => {
    refreshDatasets().catch((reason: unknown) => setError(errorMessage(reason)));
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
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark">Y</div>
        <div><strong>YoloWebAgentStarter</strong><span className="brand-subtitle">local dataset workspace</span></div>
        <span className="scope-badge">Community · detect / segment</span>
      </header>
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
          busy={busy}
        />
      ) : view === "training" && displayedDataset ? (
        <TrainingView dataset={displayedDataset} onBack={() => setView("workspace")} />
      ) : (
        <Workspace
          datasets={datasets}
          selected={displayedDataset}
          classes={classes}
          images={images}
          report={report}
          activeClassId={activeClassId}
          busy={busy}
          onSelect={(dataset) => run(() => loadDataset(dataset))}
          onCreate={(name, taskType) => run(async () => { const dataset = await api.createDataset(name, taskType); await refreshDatasets(); await loadDataset(dataset); setNotice("数据集已创建"); })}
          onAddClass={(name) => selected && run(async () => { const item = await api.createClass(selected.id, name); setClasses((items) => [...items, item]); setActiveClassId(item.id); setNotice("类别已添加"); })}
          onUpload={(files, split) => selected && run(async () => { const result = await api.uploadImages(selected.id, files, split); setImages((items) => [...items, ...result.items]); await refreshDatasets(); setNotice(`已导入 ${result.imported} 张图片`); })}
          onScan={(path, split) => selected && run(async () => { const result = await api.scanImages(selected.id, path, split); const nextImages = await api.listImages(selected.id); setImages(nextImages.items); await refreshDatasets(); setNotice(`扫描完成：导入 ${result.imported} 张，跳过 ${result.skipped} 张`); })}
          onImport={(file, name, taskType) => run(async () => { const result = await api.importYolo(file, name, taskType); await refreshDatasets(); await loadDataset(result.dataset); setNotice(`已导入 ${result.imported_images} 张图片和 ${result.imported_annotations} 个标注`); })}
          onValidate={() => selected && run(async () => setReport(await api.validateDataset(selected.id)))}
          onActiveClassChange={setActiveClassId}
          onOpenImage={openImage}
          onOpenTraining={() => setView("training")}
          onSplitChange={(image, split) => selected && run(async () => { const updated = await api.updateImageSplit(selected.id, image.id, split); setImages((items) => items.map((item) => item.id === image.id ? updated : item)); await refreshDatasets(); })}
        />
      )}
    </div>
  );
}

function Workspace(props: {
  datasets: Dataset[]; selected?: Dataset; classes: ClassLabel[]; images: ImageItem[]; report?: ValidationReport; activeClassId: string; busy: boolean;
  onSelect: (dataset: Dataset) => void; onCreate: (name: string, type: TaskType) => void; onAddClass: (name: string) => void;
  onUpload: (files: File[], split: SplitName) => void; onScan: (path: string, split: SplitName) => void; onImport: (file: File, name: string, type: TaskType) => void; onValidate: () => void;
  onActiveClassChange: (id: string) => void; onOpenImage: (image: ImageItem) => void; onOpenTraining: () => void; onSplitChange: (image: ImageItem, split: SplitName) => void;
}) {
  const [name, setName] = useState("");
  const [type, setType] = useState<TaskType>("detect");
  const [className, setClassName] = useState("");
  const [split, setSplit] = useState<SplitName>("train");
  const [scanPath, setScanPath] = useState("");
  const [importName, setImportName] = useState("Imported dataset");
  const [importType, setImportType] = useState<TaskType>("detect");
  const uploadRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<SplitName | "all">("all");
  const visibleImages = useMemo(() => tab === "all" ? props.images : props.images.filter((image) => image.split === tab), [props.images, tab]);

  return (
    <main className="content-grid">
      <aside className="sidebar panel">
        <div className="panel-heading"><div><span className="eyebrow">PROJECTS</span><h2>数据集</h2></div><span className="count-badge">{props.datasets.length}</span></div>
        <div className="dataset-list">
          {props.datasets.map((dataset) => <button key={dataset.id} className={props.selected?.id === dataset.id ? "dataset-item selected" : "dataset-item"} onClick={() => props.onSelect(dataset)}><span>{dataset.name}</span><small>{dataset.task_type} · {dataset.image_count} imgs</small></button>)}
          {!props.datasets.length && <p className="muted">还没有数据集。</p>}
        </div>
        <div className="sidebar-form"><span className="eyebrow">NEW DATASET</span><input placeholder="数据集名称" value={name} onChange={(event) => setName(event.target.value)} /><div className="inline-fields"><select value={type} onChange={(event) => setType(event.target.value as TaskType)}><option value="detect">Detect / BBox</option><option value="segment">Segment / Polygon</option></select><button className="button primary" disabled={!name.trim() || props.busy} onClick={() => { props.onCreate(name.trim(), type); setName(""); }}>创建</button></div></div>
      </aside>
      <section className="workspace panel">
        {!props.selected ? <EmptyWorkspace onOpenImport={() => importRef.current?.click()} importName={importName} setImportName={setImportName} importType={importType} setImportType={setImportType} /> : <>
          <div className="workspace-header"><div><span className="eyebrow">DATASET WORKSPACE</span><h1>{props.selected.name}</h1><p>{props.selected.task_type === "detect" ? "Bounding-box detection" : "Polygon segmentation"} · {props.selected.image_count} 张图片 · {props.selected.class_count} 个类别</p></div><div className="header-actions"><button className="button primary" onClick={props.onOpenTraining}>训练</button><a className="button" href={api.exportYoloUrl(props.selected.id)}>导出 YOLO ZIP</a></div></div>
          <div className="action-row"><input ref={uploadRef} type="file" accept="image/*" multiple hidden onChange={(event) => { if (event.target.files) props.onUpload(Array.from(event.target.files), split); event.currentTarget.value = ""; }} /><select value={split} onChange={(event) => setSplit(event.target.value as SplitName)}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select><button className="button primary" onClick={() => uploadRef.current?.click()}>上传图片</button><input placeholder="扫描受管目录相对路径" value={scanPath} onChange={(event) => setScanPath(event.target.value)} /><button className="button" disabled={!scanPath.trim()} onClick={() => props.onScan(scanPath.trim(), split)}>扫描目录</button><button className="button" onClick={props.onValidate}>运行校验</button></div>
          <div className="split-tabs">{(["all", "train", "val", "test"] as const).map((item) => <button key={item} className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>{item === "all" ? "全部" : item}</button>)}</div>
          <div className="class-strip"><span className="eyebrow">CLASSES</span>{props.classes.map((item) => <button key={item.id} className={props.activeClassId === item.id ? "class-chip active" : "class-chip"} onClick={() => props.onActiveClassChange(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}<input value={className} placeholder="新增类别" onChange={(event) => setClassName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && className.trim()) { props.onAddClass(className.trim()); setClassName(""); } }} /></div>
          {props.report && <ValidationPanel report={props.report} />}
          <div className="image-grid">{visibleImages.map((image) => <button className="image-card" key={image.id} onClick={() => props.onOpenImage(image)}><img src={apiUrl(image.file_url)} alt={image.file_name} /><div><strong>{image.file_name}</strong><span>{image.status === "annotated" ? "已标注" : "未标注"}</span><select value={image.split} onClick={(event) => event.stopPropagation()} onChange={(event) => { event.stopPropagation(); props.onSplitChange(image, event.target.value as SplitName); }}><option value="train">train</option><option value="val">val</option><option value="test">test</option></select></div></button>)}{!visibleImages.length && <div className="empty-state"><strong>开始导入图片</strong><span>上传图片后，在这里选择一张进入标注工作台。</span></div>}</div>
        </>}
      </section>
      <input ref={importRef} type="file" accept=".zip" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) props.onImport(file, importName, importType); event.currentTarget.value = ""; }} />
    </main>
  );
}

function EmptyWorkspace(props: { onOpenImport: () => void; importName: string; setImportName: (value: string) => void; importType: TaskType; setImportType: (value: TaskType) => void }) {
  return <div className="empty-workspace"><div className="empty-orbit">◎</div><span className="eyebrow">GET STARTED</span><h1>建立你的第一个数据集</h1><p>从图片开始，完成类别、标注、校验，再导出标准 YOLO 数据集。</p><div className="empty-actions"><button className="button primary" onClick={props.onOpenImport}>导入 YOLO ZIP</button><input value={props.importName} onChange={(event) => props.setImportName(event.target.value)} placeholder="导入数据集名称" /><select value={props.importType} onChange={(event) => props.setImportType(event.target.value as TaskType)}><option value="detect">detect</option><option value="segment">segment</option></select></div></div>;
}

function ValidationPanel({ report }: { report: ValidationReport }) {
  return <div className={report.valid ? "validation valid" : "validation invalid"}><strong>{report.valid ? "校验通过" : "发现需要处理的问题"}</strong><span>{report.error_count} errors · {report.warning_count} warnings</span>{report.issues.slice(0, 3).map((issue) => <small key={`${issue.code}-${issue.image_id ?? "dataset"}`}>{issue.level === "error" ? "!" : "·"} {issue.message}</small>)}</div>;
}

function AnnotationView(props: { dataset: Dataset; image: ImageItem; classes: ClassLabel[]; annotations: Annotation[]; activeClassId: string; onClassChange: (id: string) => void; onBack: () => void; onSave: (drafts: AnnotationDraft[]) => void; busy: boolean }) {
  const [drafts, setDrafts] = useState<AnnotationDraft[]>(props.annotations.map((item) => ({ id: item.id, class_id: item.class_id, type: item.type, bbox: item.bbox ?? undefined, polygon: item.polygon ?? undefined, source: item.source })));
  useEffect(() => setDrafts(props.annotations.map((item) => ({ id: item.id, class_id: item.class_id, type: item.type, bbox: item.bbox ?? undefined, polygon: item.polygon ?? undefined, source: item.source }))), [props.annotations]);
  const annotated = drafts.map((draft, index) => ({ ...draft, id: draft.id ?? `draft-${index}`, image_id: props.image.id, dataset_id: props.dataset.id, class_index: 0, label: "", color: props.classes.find((item) => item.id === draft.class_id)?.color ?? "#f97316", created_at: "", updated_at: "" })) as Annotation[];
  return <main className="annotation-layout"><div className="annotation-head"><button className="button" onClick={props.onBack}>← 返回数据集</button><div><span className="eyebrow">{props.dataset.name} / {props.dataset.task_type}</span><h1>{props.image.file_name}</h1></div><div className="annotation-actions"><select value={props.activeClassId} onChange={(event) => props.onClassChange(event.target.value)}>{props.classes.map((item) => <option key={item.id} value={item.id}>{item.class_index}: {item.name}</option>)}</select><button className="button primary" disabled={props.busy || !props.activeClassId} onClick={() => props.onSave(drafts)}>保存标注</button></div></div><div className="annotation-body"><AnnotationCanvas image={props.image} classes={props.classes} annotations={annotated} activeClassId={props.activeClassId} onChange={setDrafts} /><aside className="annotation-sidebar panel"><span className="eyebrow">SAVED SHAPES</span><h2>{drafts.length} 个标注</h2>{drafts.map((draft, index) => <div className="shape-row" key={draft.id ?? index}><span className="shape-index">{index + 1}</span><div><strong>{props.classes.find((item) => item.id === draft.class_id)?.name ?? "Unknown"}</strong><small>{draft.type === "bbox" ? "Bounding box" : "Polygon"}</small></div><button className="icon-button" onClick={() => setDrafts((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}{!drafts.length && <p className="muted">在图片上拖拽或点击创建标注。</p>}</aside></div></main>;
}

function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
