import { useEffect, useRef, useState, type ReactNode } from "react";
import { AnnotationCanvas, type AnnotationDraft } from "./annotation/AnnotationCanvas";
import { api, apiUrl } from "./api/client";
import { StarterShell, type StarterSection } from "./components/StarterShell";
import { TrainingView } from "./pages/TrainingView";
import { ModelsView } from "./pages/ModelsView";
import { SettingsView } from "./pages/SettingsView";
import { LogsView } from "./pages/LogsView";
import { readLocale, saveLocale, type AppLocale } from "./locale";
import type { Annotation, BBox, ClassLabel, Dataset, DatasetQualityReport, DuplicateReport, ImageItem, SamCapabilities, SamPrediction, SplitName, TaskType, ValidationReport } from "./types";

type View = "workspace" | "annotation" | "training" | "models" | "settings-sam" | "settings-language" | "logs";
type CardReportKind = "validation" | "quality" | "duplicates";
const ANNOTATION_IMAGE_PAGE_SIZE = 20;

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset>();
  const [classes, setClasses] = useState<ClassLabel[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageItem>();
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [view, setView] = useState<View>("workspace");
  const [activeClassId, setActiveClassId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [samCapabilities, setSamCapabilities] = useState<SamCapabilities>();
  const [locale, setLocale] = useState<AppLocale>(() => readLocale());

  const refreshDatasets = async () => {
    const result = await api.listDatasets();
    setDatasets(result);
    if (result.length > 0) setNotice("");
    if (selected) {
      const current = result.find((dataset) => dataset.id === selected.id);
      if (current) setSelected(current);
    }
  };

  const loadDataset = async (dataset: Dataset, nextView: View = "workspace") => {
    setNotice("");
    setSelected(dataset);
    setView(nextView);
    setSelectedImage(undefined);
    const [nextClasses, nextImages] = await Promise.all([api.listClasses(dataset.id), api.listImages(dataset.id)]);
    setClasses(nextClasses);
    setImages(nextImages.items);
    setActiveClassId((current) => current || nextClasses[0]?.id || "");
  };

  useEffect(() => {
    saveLocale(locale);
  }, [locale]);

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
      await loadDataset(dataset);
      const nextImages = await api.listImages(dataset.id);
      const image = nextImages.items.find((item) => item.status === "unannotated") ?? nextImages.items[0];
      if (!image) {
        setNotice("该数据集还没有图片可标注。");
        return;
      }
      setSelectedImage(image);
      setAnnotations(await api.getAnnotations(dataset.id, image.id));
      setView("annotation");
    });
  };

  const displayedDataset = selected;
  const activeSection: StarterSection = view === "annotation" ? "workspace" : view === "settings-sam" || view === "settings-language" ? "settings" : view;
  const navigate = (section: StarterSection) => {
    setNotice("");
    setError("");
    if ((section === "training" || section === "models") && !selected && datasets[0]) {
      void loadDataset(datasets[0], section);
      return;
    }
    const globalSection = section === "settings" || section === "logs";
    if (!globalSection && section !== "workspace" && !selected) {
      setNotice("请先创建或选择一个数据集。");
      return;
    }
    setView(section === "settings" ? "settings-sam" : section);
  };

  return (
    <StarterShell active={activeSection} datasetName={selected?.name} onNavigate={navigate} locale={locale}>
      {error && <div className="toast error">{error}</div>}
      {notice && <div className="toast success">{notice}</div>}
      {view === "annotation" && displayedDataset && selectedImage ? (
        <AnnotationView
          dataset={displayedDataset}
          image={selectedImage}
          images={images}
          onImageSelect={(image) => void openImage(image)}
          classes={classes}
          annotations={annotations}
          activeClassId={activeClassId}
          onClassChange={setActiveClassId}
          onBack={() => setView("workspace")}
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
        <TrainingView datasets={datasets} dataset={displayedDataset} onDatasetChange={(dataset) => void loadDataset(dataset, "training")} onOpenModels={() => setView("models")} />
      ) : view === "models" && displayedDataset ? (
        <ModelsView dataset={displayedDataset} />
      ) : view === "settings-sam" ? (
        <SettingsView tab="sam" locale={locale} onLocaleChange={setLocale} onTabChange={(tab) => setView(tab === "sam" ? "settings-sam" : "settings-language")} onSamSettingsChange={() => { api.getSystemInfo().then((info) => setSamCapabilities(info.sam)).catch(() => undefined); }} />
      ) : view === "settings-language" ? (
        <SettingsView tab="language" locale={locale} onLocaleChange={setLocale} onTabChange={(tab) => setView(tab === "sam" ? "settings-sam" : "settings-language")} />
      ) : view === "logs" ? (
        <LogsView locale={locale} />
      ) : (
        <DatasetHome
          datasets={datasets}
          busy={busy}
          onCreate={(name, taskType) => run(async () => { const dataset = await api.createDataset(name, taskType); await refreshDatasets(); await loadDataset(dataset); setNotice("数据集已创建"); })}
          onImport={(file, name, taskType, format) => run(async () => { const result = format === "coco" ? await api.importCoco(file, name, taskType) : await api.importYolo(file, name, taskType); await refreshDatasets(); await loadDataset(result.dataset); setNotice(`已导入 ${result.imported_images} 张图片和 ${result.imported_annotations} 个标注`); })}
          onValidate={(dataset) => runResult(() => api.validateDataset(dataset.id))}
          onQuality={(dataset) => runResult(() => api.qualityReport(dataset.id))}
          onContinueAnnotation={(dataset) => void continueAnnotation(dataset)}
          onUpload={(dataset, files, split) => run(async () => {
            const result = await api.uploadImages(dataset.id, files, split);
            await refreshDatasets();
            if (selected?.id === dataset.id) {
              const nextImages = await api.listImages(dataset.id);
              setImages(nextImages.items);
            }
            setNotice(`已添加 ${result.imported} 张图片到 ${split}`);
          })}
          onTrain={(dataset) => void run(() => loadDataset(dataset, "training"))}
          onDuplicates={(dataset) => runResult(() => api.duplicateReport(dataset.id))}
          onDelete={(dataset) => run(async () => {
            await api.deleteDataset(dataset.id);
            setDatasets((items) => items.filter((item) => item.id !== dataset.id));
            if (selected?.id === dataset.id) {
              setSelected(undefined);
              setClasses([]);
              setImages([]);
              setSelectedImage(undefined);
              setAnnotations([]);
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
  onCreate: (name: string, type: TaskType) => void;
  onImport: (file: File, name: string, type: TaskType, format: "yolo" | "coco") => void;
  onValidate: (dataset: Dataset) => Promise<ValidationReport | undefined>;
  onQuality: (dataset: Dataset) => Promise<DatasetQualityReport | undefined>;
  onContinueAnnotation: (dataset: Dataset) => void;
  onUpload: (dataset: Dataset, files: File[], split: SplitName) => void;
  onTrain: (dataset: Dataset) => void;
  onDuplicates: (dataset: Dataset) => Promise<DuplicateReport | undefined>;
  onDelete: (dataset: Dataset) => void;
}) {
  const [dialog, setDialog] = useState<"create" | "import" | null>(null);
  const [exportTarget, setExportTarget] = useState<Dataset>();
  const [exportFormat, setExportFormat] = useState<"yolo" | "coco">("yolo");
  const [deleteTarget, setDeleteTarget] = useState<Dataset>();
  const [uploadTarget, setUploadTarget] = useState<Dataset>();
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadSplit, setUploadSplit] = useState<SplitName>("train");
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [archive, setArchive] = useState<File>();
  const [format, setFormat] = useState<"yolo" | "coco">("yolo");
  const [draggingArchive, setDraggingArchive] = useState(false);
  const [validationReports, setValidationReports] = useState<Record<string, ValidationReport>>({});
  const [qualityReports, setQualityReports] = useState<Record<string, DatasetQualityReport>>({});
  const [duplicateReports, setDuplicateReports] = useState<Record<string, DuplicateReport>>({});
  const [expandedReports, setExpandedReports] = useState<Record<string, CardReportKind | undefined>>({});
  const fileRef = useRef<HTMLInputElement>(null);
  const uploadFileRef = useRef<HTMLInputElement>(null);
  const [draggingImages, setDraggingImages] = useState(false);

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
  const chooseArchive = (file?: File) => {
    if (file) setArchive(file);
  };
  const closeUpload = () => {
    setUploadTarget(undefined);
    setUploadFiles([]);
    setUploadSplit("train");
    setDraggingImages(false);
  };
  const chooseImages = (files?: FileList | File[]) => {
    if (files) setUploadFiles(Array.from(files));
  };
  const startUpload = () => {
    if (!uploadTarget || !uploadFiles.length) return;
    props.onUpload(uploadTarget, uploadFiles, uploadSplit);
    closeUpload();
  };
  const openExport = (dataset: Dataset) => {
    setExportTarget(dataset);
    setExportFormat("yolo");
  };
  const exportUrl = exportTarget && (exportFormat === "coco" ? api.exportCocoUrl(exportTarget.id) : api.exportYoloUrl(exportTarget.id));
  const isReportOpen = (datasetId: string, kind: CardReportKind) => expandedReports[datasetId] === kind;
  const closeReport = (datasetId: string) => setExpandedReports((items) => ({ ...items, [datasetId]: undefined }));
  const openReport = (datasetId: string, kind: CardReportKind) => setExpandedReports((items) => ({ ...items, [datasetId]: kind }));
  const toggleValidation = async (dataset: Dataset) => {
    if (isReportOpen(dataset.id, "validation")) return closeReport(dataset.id);
    if (validationReports[dataset.id]) return openReport(dataset.id, "validation");
    const report = await props.onValidate(dataset);
    if (report) { setValidationReports((items) => ({ ...items, [dataset.id]: report })); openReport(dataset.id, "validation"); }
  };
  const toggleQuality = async (dataset: Dataset) => {
    if (isReportOpen(dataset.id, "quality")) return closeReport(dataset.id);
    if (qualityReports[dataset.id]) return openReport(dataset.id, "quality");
    const report = await props.onQuality(dataset);
    if (report) { setQualityReports((items) => ({ ...items, [dataset.id]: report })); openReport(dataset.id, "quality"); }
  };
  const toggleDuplicates = async (dataset: Dataset) => {
    if (isReportOpen(dataset.id, "duplicates")) return closeReport(dataset.id);
    if (duplicateReports[dataset.id]) return openReport(dataset.id, "duplicates");
    const report = await props.onDuplicates(dataset);
    if (report) { setDuplicateReports((items) => ({ ...items, [dataset.id]: report })); openReport(dataset.id, "duplicates"); }
  };

  return <main className="dataset-home-page">
    <header className="dataset-home-header">
      <div><h1>数据集</h1><p>在这里创建、导入和管理你的 YOLO 数据集。</p></div>
      <div className="header-actions"><button className="button" onClick={() => setDialog("import")}>导入数据集</button><button className="button primary" onClick={() => setDialog("create")}>新建数据集</button></div>
    </header>
    <section className="dataset-overview">
      <div className="section-title-row"><div><span className="eyebrow">ALL DATASETS</span><h2>所有数据集</h2></div><span className="dataset-total">{props.datasets.length} 个数据集</span></div>
      <div className="dataset-grid">
        {props.datasets.map((dataset, index) => <article key={dataset.id} className="dataset-card">
          <header><span className={`dataset-icon tone-${index % 3}`}>{dataset.task_type === "segment" || dataset.task_type === "obb" ? "◇" : dataset.task_type === "classify" ? "○" : "□"}</span><div><h2>{dataset.name}</h2><small>{taskDescription(dataset.task_type)}</small></div><div className="dataset-card-header-actions"><span className="status ready">就绪</span><button className="dataset-add-image-action" disabled={props.busy} onClick={() => setUploadTarget(dataset)}>+ 添加图片</button></div></header>
          <div className="dataset-metrics"><span><small>图片</small><strong>{dataset.image_count.toLocaleString()}</strong></span><span><small>类别</small><strong>{dataset.class_count.toLocaleString()}</strong></span></div>
          <div className="dataset-progress"><span>标注进度</span><strong>{dataset.image_count ? `${Math.round(dataset.annotated_image_count / dataset.image_count * 100)}%` : "等待导入"}</strong></div>
          <div className="progress-track"><i style={{ width: `${dataset.image_count ? dataset.annotated_image_count / dataset.image_count * 100 : 0}%` }} /></div>
          <button className="button primary dataset-primary-action" disabled={props.busy || !dataset.image_count} onClick={() => props.onContinueAnnotation(dataset)}>继续标注</button>
          <div className="dataset-card-actions">
            <button className="button" disabled={props.busy} onClick={() => props.onTrain(dataset)}>训练</button>
            <button className="button" disabled={props.busy || !dataset.image_count} onClick={() => openExport(dataset)}>导出数据集</button>
            <button className="button" disabled={props.busy || !dataset.image_count} onClick={() => void toggleDuplicates(dataset)}>{isReportOpen(dataset.id, "duplicates") ? "收起重复图" : "重复/相似图"}</button>
            <button className="button" disabled={props.busy} onClick={() => void toggleValidation(dataset)}>{isReportOpen(dataset.id, "validation") ? "收起校验" : "运行校验"}</button>
            <button className="button" disabled={props.busy} onClick={() => void toggleQuality(dataset)}>{isReportOpen(dataset.id, "quality") ? "收起报告" : "质量报告"}</button>
            <button className="button dataset-delete-action" disabled={props.busy} onClick={() => setDeleteTarget(dataset)}>删除</button>
          </div>
          {isReportOpen(dataset.id, "validation") && validationReports[dataset.id] && <CardReport onClose={() => closeReport(dataset.id)}><ValidationPanel report={validationReports[dataset.id]} compact /></CardReport>}
          {isReportOpen(dataset.id, "quality") && qualityReports[dataset.id] && <CardReport onClose={() => closeReport(dataset.id)}><QualityPanel report={qualityReports[dataset.id]} compact /></CardReport>}
          {isReportOpen(dataset.id, "duplicates") && duplicateReports[dataset.id] && <CardReport onClose={() => closeReport(dataset.id)}><DuplicatePanel report={duplicateReports[dataset.id]} compact /></CardReport>}
        </article>)}
        {!props.datasets.length && <section className="dataset-empty-card"><span className="dataset-empty-icon">□</span><h2>还没有数据集</h2><p>新建一个空数据集，或导入已有的 YOLO ZIP 数据集。</p><div><button className="button" onClick={() => setDialog("import")}>导入数据</button><button className="button primary" onClick={() => setDialog("create")}>新建数据集</button></div></section>}
      </div>
    </section>
    {dialog && <div className="modal-backdrop data-exchange-backdrop" role="presentation" onMouseDown={closeDialog}>
      <section className="dataset-dialog data-exchange-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="data-exchange-header"><div><span className="eyebrow">{dialog === "create" ? "NEW DATASET" : "IMPORT CENTER"}</span><h2 id="dataset-dialog-title">{dialog === "create" ? "新建数据集" : "导入数据集"}</h2><p>{dialog === "create" ? "创建后即可上传图片并开始标注。" : "选择格式、任务类型和压缩包，将已有标注导入本地工作区。"}</p></div><button className="icon-button" onClick={closeDialog} aria-label="关闭">×</button></header>
        {dialog === "import" && <div className="exchange-steps" aria-label="导入步骤"><span className="active">1 选择格式</span><i /><span className="active">2 上传文件</span><i /><span>3 开始导入</span></div>}
        {dialog === "import" && <section><span className="exchange-label">数据集格式</span><div className="format-option-grid"><button className={format === "yolo" ? "format-option selected" : "format-option"} onClick={() => setFormat("yolo")}><strong>YOLO ZIP</strong><span>data.yaml、images 与 labels</span></button><button className={format === "coco" ? "format-option selected" : "format-option"} onClick={() => setFormat("coco")}><strong>COCO ZIP</strong><span>annotations.json 与 images</span></button></div></section>}
        <div className="exchange-fields"><label>数据集名称<input value={name} placeholder={dialog === "import" ? "例如：road-signs" : "例如：my-dataset"} onChange={(event) => setName(event.target.value)} autoFocus /></label><label>任务类型<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}><option value="detect">目标检测（Bounding Box）</option><option value="segment">实例分割（Polygon / SAM）</option><option value="obb">旋转框（OBB）</option><option value="classify">图像分类</option></select></label></div>
        {dialog === "import" && <><input ref={fileRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => chooseArchive(event.target.files?.[0])} /><button className={draggingArchive ? "exchange-dropzone dragging" : "exchange-dropzone"} onClick={() => fileRef.current?.click()} onDragEnter={(event) => { event.preventDefault(); setDraggingArchive(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDraggingArchive(false)} onDrop={(event) => { event.preventDefault(); setDraggingArchive(false); chooseArchive(event.dataTransfer.files[0]); }}><span className="exchange-dropzone-icon">⇧</span><strong>{archive?.name ?? `拖放或选择 ${format.toUpperCase()} ZIP 文件`}</strong><small>{archive ? `${Math.ceil(archive.size / 1024)} KB · 已准备导入` : format === "yolo" ? "ZIP 内应包含 data.yaml、images 和 labels" : "ZIP 内应包含 annotations.json/instances.json 与 images"}</small></button></>}
        <footer className="data-exchange-footer"><span>{dialog === "import" ? "文件仅导入到本机受管数据目录。" : "创建后可从卡片继续导入图片或已有数据集。"}</span><div><button className="button" onClick={closeDialog}>取消</button><button className="button primary" disabled={props.busy || !name.trim() || (dialog === "import" && !archive)} onClick={dialog === "create" ? create : importArchive}>{dialog === "create" ? "创建数据集" : "开始导入"}</button></div></footer>
      </section>
    </div>}
    {uploadTarget && <div className="modal-backdrop data-exchange-backdrop" role="presentation" onMouseDown={closeUpload}>
      <section className="dataset-dialog data-exchange-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-upload-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="data-exchange-header"><div><span className="eyebrow">ADD IMAGES</span><h2 id="dataset-upload-dialog-title">添加图片到“{uploadTarget.name}”</h2><p>选择多张图片并指定 split，图片会进入当前本地数据集。</p></div><button className="icon-button" onClick={closeUpload} aria-label="关闭">×</button></header>
        <div className="upload-options"><label>导入到 split<select value={uploadSplit} onChange={(event) => setUploadSplit(event.target.value as SplitName)}><option value="train">train · 训练集</option><option value="val">val · 验证集</option><option value="test">test · 测试集</option></select></label></div>
        <input ref={uploadFileRef} type="file" accept="image/*" multiple hidden onChange={(event) => { chooseImages(event.target.files ?? undefined); event.currentTarget.value = ""; }} />
        <button className={draggingImages ? "exchange-dropzone dragging" : "exchange-dropzone"} onClick={() => uploadFileRef.current?.click()} onDragEnter={(event) => { event.preventDefault(); setDraggingImages(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDraggingImages(false)} onDrop={(event) => { event.preventDefault(); setDraggingImages(false); chooseImages(event.dataTransfer.files); }}><span className="exchange-dropzone-icon">⇧</span><strong>{uploadFiles.length ? `已选择 ${uploadFiles.length} 张图片` : "拖放或选择图片"}</strong><small>{uploadFiles.length ? uploadFiles.slice(0, 3).map((file) => file.name).join("、") + (uploadFiles.length > 3 ? ` 等 ${uploadFiles.length} 张` : "") : "支持 JPG、PNG、WEBP 等常见图片格式，可多选"}</small></button>
        <footer className="data-exchange-footer"><span>原图会复制到受管数据目录，现有标注不会受影响。</span><div><button className="button" onClick={closeUpload}>取消</button><button className="button primary" disabled={props.busy || !uploadFiles.length} onClick={startUpload}>添加 {uploadFiles.length || ""} 张图片</button></div></footer>
      </section>
    </div>}
    {exportTarget && <div className="modal-backdrop data-exchange-backdrop" role="presentation" onMouseDown={() => setExportTarget(undefined)}>
      <section className="dataset-dialog data-exchange-dialog" role="dialog" aria-modal="true" aria-labelledby="dataset-export-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <header className="data-exchange-header"><div><span className="eyebrow">EXPORT CENTER</span><h2 id="dataset-export-dialog-title">导出“{exportTarget.name}”</h2><p>导出会使用当前已保存的图片 split、类别和标注。</p></div><button className="icon-button" onClick={() => setExportTarget(undefined)} aria-label="关闭">×</button></header>
        <div className="exchange-summary"><span>任务类型<strong>{taskDescription(exportTarget.task_type)}</strong></span><span>图片<strong>{exportTarget.image_count} 张</strong></span><span>已标注<strong>{exportTarget.annotated_image_count} 张</strong></span></div>
        <section><span className="exchange-label">导出格式</span><div className="format-option-grid"><button className={exportFormat === "yolo" ? "format-option selected" : "format-option"} onClick={() => setExportFormat("yolo")}><strong>YOLO ZIP</strong><span>标准 YOLO 目录、标签与 data.yaml</span></button>{["detect", "segment"].includes(exportTarget.task_type) && <button className={exportFormat === "coco" ? "format-option selected" : "format-option"} onClick={() => setExportFormat("coco")}><strong>COCO ZIP</strong><span>annotations.json 与原始图片</span></button>}</div></section>
        <footer className="data-exchange-footer"><span>下载文件会由浏览器保存到默认下载目录。</span><div><button className="button" onClick={() => setExportTarget(undefined)}>取消</button><a className="button primary" href={exportUrl ?? undefined} onClick={() => setExportTarget(undefined)}>下载 {exportFormat.toUpperCase()} ZIP</a></div></footer>
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

function CardReport({ children, onClose }: { children: ReactNode; onClose: () => void }) { return <div className="dataset-card-report"><button className="dataset-card-report-close" onClick={onClose}>收起结果</button>{children}</div>; }
function QualityPanel({ report, compact = false }: { report: DatasetQualityReport; compact?: boolean }) { return <section className={compact ? "quality-panel compact-quality-panel" : "quality-panel"}><div className="quality-summary"><span><small>标注覆盖率</small><strong>{Math.round(report.summary.coverage * 100)}%</strong></span><span><small>标注总数</small><strong>{report.summary.annotation_count}</strong></span><span><small>小目标</small><strong>{report.summary.small_object_count}</strong></span><span><small>问题</small><strong>{report.issues.length}</strong></span></div>{!compact && <div className="quality-grid"><div><h3>类别分布</h3>{report.class_distribution.map((item) => <p key={item.class_id}><span>{item.class_index}. {item.name}</span><b>{item.count}</b><i><em style={{ width: `${Math.max(item.ratio * 100, item.count ? 3 : 0)}%` }} /></i></p>)}</div><div><h3>需要关注</h3>{report.issues.slice(0, 5).map((item, index) => <p key={`${item.type}-${index}`}><b>{item.level}</b><span>{item.message}</span></p>)}{!report.issues.length && <p>当前未发现需要关注的问题。</p>}</div></div>}</section>; }
function DuplicatePanel({ report, compact = false }: { report: DuplicateReport; compact?: boolean }) { return <section className={compact ? "quality-panel compact-quality-panel" : "quality-panel"}><div className="quality-summary"><span><small>精确重复</small><strong>{report.duplicate}</strong></span><span><small>相似图片</small><strong>{report.similar}</strong></span><span><small>无效图片</small><strong>{report.invalid_images}</strong></span><span><small>分析图片</small><strong>{report.images}</strong></span></div>{!compact && <div className="quality-grid"><div><h3>相似性分组</h3>{report.groups.slice(0, 6).map((item, index) => <p key={index}><span>{item.kind === "exact" ? "精确重复" : "相似"} · {item.image_ids.length} 张</span><b>{(item.score * 100).toFixed(0)}%</b></p>)}{!report.groups.length && <p>未发现重复或相似图片。</p>}</div><div><h3>处理建议</h3><p>重复检测为只读报告，不会删除任何图片。</p><p>相似图应由你在标注和 split 前人工决定是否保留。</p></div></div>}</section>; }


function ValidationPanel({ report, compact = false }: { report: ValidationReport; compact?: boolean }) {
  return <div className={`${report.valid ? "validation valid" : "validation invalid"}${compact ? " compact-validation" : ""}`}><strong>{report.valid ? "校验通过" : "发现需要处理的问题"}</strong><span>{report.error_count} errors · {report.warning_count} warnings</span>{!compact && report.issues.slice(0, 3).map((issue) => <small key={`${issue.code}-${issue.image_id ?? "dataset"}`}>{issue.level === "error" ? "!" : "·"} {issue.message}</small>)}</div>;
}

export function AnnotationView(props: { dataset: Dataset; image: ImageItem; images?: ImageItem[]; onImageSelect?: (image: ImageItem) => void; classes: ClassLabel[]; annotations: Annotation[]; activeClassId: string; onClassChange: (id: string) => void; onBack: () => void; onPrevious?: () => void; onNext?: () => void; hasPrevious?: boolean; hasNext?: boolean; onSave: (drafts: AnnotationDraft[]) => void; onSam: (box: BBox) => Promise<SamPrediction>; onSamPoints: (points: [number, number][]) => Promise<SamPrediction>; busy: boolean; samCapabilities?: SamCapabilities }) {
  const toDraft = (item: Annotation): AnnotationDraft => ({ id: item.id, class_id: item.class_id, type: item.type, bbox: item.bbox ?? undefined, polygon: item.polygon ?? undefined, obb: item.obb ?? undefined, source: item.source });
  const [drafts, setDrafts] = useState<AnnotationDraft[]>(props.annotations.map(toDraft));
  const [imagePage, setImagePage] = useState(0);
  useEffect(() => setDrafts(props.annotations.map(toDraft)), [props.annotations, props.image.id]);
  const imageCount = props.images?.length ?? 0;
  const imagePageCount = Math.max(1, Math.ceil(imageCount / ANNOTATION_IMAGE_PAGE_SIZE));
  useEffect(() => {
    if (!props.images?.length) {
      setImagePage(0);
      return;
    }
    const imageIndex = props.images.findIndex((item) => item.id === props.image.id);
    if (imageIndex >= 0) setImagePage(Math.min(Math.floor(imageIndex / ANNOTATION_IMAGE_PAGE_SIZE), imagePageCount - 1));
  }, [props.image.id, props.images, imagePageCount]);
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
  useEffect(() => { const onKey = (event: KeyboardEvent) => { if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return; if (event.key === "ArrowLeft" && props.hasPrevious && !props.busy) props.onPrevious?.(); if (event.key === "ArrowRight" && props.hasNext && !props.busy) props.onNext?.(); if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") { event.preventDefault(); props.onSave(drafts); } }; window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, [drafts, props]);
  return <main className="annotation-layout"><div className="annotation-head"><button className="button" onClick={props.onBack}>← 返回数据集</button><div><span className="eyebrow">{props.dataset.name} / {props.dataset.task_type}</span><h1>{props.image.file_name}</h1><p className="muted">←/→ 切换图片 · ⌘/Ctrl+S 保存</p></div><div className="annotation-navigation" aria-label="图片导航">{props.onPrevious && <button className="button" disabled={props.busy || !props.hasPrevious} title="上一张（←）" onClick={props.onPrevious}>← 上一张</button>}{props.onNext && <button className="button" disabled={props.busy || !props.hasNext} title="下一张（→）" onClick={props.onNext}>下一张 →</button>}</div><div className="annotation-actions"><select value={props.activeClassId} onChange={(event) => props.onClassChange(event.target.value)}>{props.classes.map((item) => <option key={item.id} value={item.id}>{item.class_index}: {item.name}</option>)}</select><button className="button primary" disabled={props.busy || !props.activeClassId} onClick={() => props.onSave(drafts)}>保存标注</button></div></div><div className="annotation-body">
    {props.images && props.images.length > 0 && <aside className="annotation-image-sidebar panel"><div className="annotation-list-heading"><span className="eyebrow">DATASET IMAGES</span><strong>{props.images.length} 张图片</strong></div><div className="annotation-image-list">{props.images.slice(imagePage * ANNOTATION_IMAGE_PAGE_SIZE, (imagePage + 1) * ANNOTATION_IMAGE_PAGE_SIZE).map((item, index) => { const absoluteIndex = imagePage * ANNOTATION_IMAGE_PAGE_SIZE + index; return <button key={item.id} className={item.id === props.image.id ? "annotation-image-item active" : "annotation-image-item"} aria-label={`选择图片 ${item.file_name}`} onClick={() => props.onImageSelect?.(item)}><img src={apiUrl(item.file_url)} alt="" /><span className="annotation-image-copy"><strong>{String(absoluteIndex + 1).padStart(2, "0")} · {item.file_name}</strong><small className={item.status === "annotated" ? "annotated" : "unannotated"}>{item.split} · {item.status === "annotated" ? "已标注" : "未标注"}</small></span></button>; })}</div>{imagePageCount > 1 && <div className="annotation-image-pagination"><button className="button" disabled={imagePage === 0} onClick={() => setImagePage((page) => Math.max(0, page - 1))}>上一页</button><span>第 {imagePage + 1} / {imagePageCount} 页</span><button className="button" disabled={imagePage >= imagePageCount - 1} onClick={() => setImagePage((page) => Math.min(imagePageCount - 1, page + 1))}>下一页</button></div>}</aside>}
    <div className="annotation-canvas-column">{props.dataset.task_type === "classify" ? <section className="canvas-card classification-card"><span className="eyebrow">IMAGE CLASSIFICATION</span><h2>选择这张图片的唯一类别</h2><p className="hint">分类数据集每张图片只能保存一个类别，训练时会按 train/类别名/图片 导出。</p><div className="class-strip">{props.classes.map((item) => <button key={item.id} className={drafts[0]?.class_id === item.id ? "class-chip active" : "class-chip"} onClick={() => chooseClass(item.id)}><i style={{ background: item.color }} />{item.class_index}: {item.name}</button>)}</div></section> : <AnnotationCanvas image={props.image} classes={props.classes} annotations={annotated} activeClassId={props.activeClassId} taskType={props.dataset.task_type} onChange={setDrafts} onSamBox={props.dataset.task_type === "segment" ? requestSam : undefined} onSamPoints={props.dataset.task_type === "segment" ? requestSamPoints : undefined} samBusy={props.busy} samCapabilities={props.samCapabilities} />}</div>
    <aside className="annotation-sidebar panel"><span className="eyebrow">SAVED SHAPES</span><h2>{drafts.length} 个标注</h2>{drafts.map((draft, index) => <div className="shape-row" key={draft.id ?? index}><span className="shape-index">{index + 1}</span><div><strong>{props.classes.find((item) => item.id === draft.class_id)?.name ?? "Unknown"}</strong><small>{annotationLabel(draft.type)}</small></div><button className="icon-button" onClick={() => setDrafts((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}{!drafts.length && <p className="muted">{props.dataset.task_type === "classify" ? "请选择一个类别。" : "在图片上拖拽或点击创建标注。"}</p>}</aside></div></main>;
}

function errorMessage(reason: unknown): string { return reason instanceof Error ? reason.message : "操作失败，请检查后端服务。"; }
function annotationLabel(type: Annotation["type"]): string { return ({ bbox: "Bounding box", polygon: "Polygon", obb: "Oriented bounding box", classify: "Image classification" })[type]; }
function taskDescription(taskType: TaskType): string { return ({ detect: "目标检测 · Bounding Box", segment: "实例分割 · Polygon / SAM", obb: "旋转框检测 · OBB", classify: "图像分类" })[taskType]; }
