import { useRef, useState } from "react";

import { VideoImportContent } from "./VideoImportModal";
import type { TaskType, VideoImportTask } from "../types";

type ImportSource = "yolo" | "coco" | "video";

const SOURCES: Array<{ id: ImportSource; icon: string; title: string; detail: string }> = [
  { id: "yolo", icon: "▣", title: "YOLO 数据集", detail: "ZIP · data.yaml、images、labels" },
  { id: "coco", icon: "◎", title: "COCO 数据集", detail: "ZIP · annotations.json 与 images" },
  { id: "video", icon: "▶", title: "视频文件", detail: "MP4、MOV、AVI · 预检后抽帧" },
];

export function ImportCenterModal({
  busy,
  onClose,
  onArchiveImport,
  onVideoCompleted,
}: {
  busy: boolean;
  onClose: () => void;
  onArchiveImport: (file: File, name: string, taskType: TaskType, format: "yolo" | "coco") => void;
  onVideoCompleted: (task: VideoImportTask) => Promise<void> | void;
}) {
  const [source, setSource] = useState<ImportSource>("yolo");
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [archive, setArchive] = useState<File>();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeSource = SOURCES.find((item) => item.id === source)!;

  const chooseSource = (next: ImportSource) => {
    setSource(next);
    setArchive(undefined);
    setDragging(false);
  };
  const importArchive = () => {
    if (!archive || !name.trim() || source === "video") return;
    onArchiveImport(archive, name.trim(), taskType, source);
    onClose();
  };

  return <div className="modal-backdrop data-exchange-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="dataset-dialog data-exchange-dialog import-center-dialog" role="dialog" aria-modal="true" aria-labelledby="import-center-title" onMouseDown={(event) => event.stopPropagation()}>
      <header className="data-exchange-header"><div><span className="eyebrow">IMPORT CENTER</span><h2 id="import-center-title">导入数据集</h2><p>选择数据来源。视频会先预检，确认后才在本机后台抽帧创建数据集。</p></div><button className="icon-button" onClick={onClose} aria-label="关闭">×</button></header>
      <div className="import-center-layout">
        <aside className="import-center-sources" aria-label="导入来源">
          <span className="exchange-label">导入来源</span>
          {SOURCES.map((item) => <button key={item.id} type="button" className={source === item.id ? "import-source-option selected" : "import-source-option"} onClick={() => chooseSource(item.id)}><span className="import-source-icon" aria-hidden>{item.icon}</span><span><strong>{item.title}</strong><small>{item.detail}</small></span></button>)}
        </aside>
        <section className="import-center-panel">
          {source === "video" ? <VideoImportContent onClose={onClose} onCompleted={onVideoCompleted} /> : <>
            <div className="import-center-step"><span>1</span><div><strong>{activeSource.title}</strong><p>{activeSource.detail}</p></div></div>
            <div className="exchange-fields"><label>数据集名称<input value={name} placeholder="例如：road-signs" onChange={(event) => setName(event.target.value)} autoFocus /></label><label>任务类型<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}><option value="detect">目标检测（Bounding Box）</option><option value="segment">实例分割（Polygon / SAM）</option><option value="obb">旋转框（OBB）</option><option value="classify">图像分类</option></select></label></div>
            <input ref={inputRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => { setArchive(event.target.files?.[0]); event.currentTarget.value = ""; }} />
            <button type="button" className={dragging ? "exchange-dropzone dragging" : "exchange-dropzone"} onClick={() => inputRef.current?.click()} onDragEnter={(event) => { event.preventDefault(); setDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); setArchive(event.dataTransfer.files[0]); }}><span className="exchange-dropzone-icon">⇧</span><strong>{archive?.name ?? `拖放或选择 ${source.toUpperCase()} ZIP 文件`}</strong><small>{archive ? `${Math.ceil(archive.size / 1024)} KB · 已准备导入` : source === "yolo" ? "ZIP 内应包含 data.yaml、images 和 labels" : "ZIP 内应包含 annotations.json/instances.json 与 images"}</small></button>
            <footer className="data-exchange-footer"><span>文件仅导入到本机受管数据目录。</span><div><button className="button" onClick={onClose}>取消</button><button className="button primary" disabled={busy || !name.trim() || !archive} onClick={importArchive}>开始导入</button></div></footer>
          </>}
        </section>
      </div>
    </section>
  </div>;
}
