import { useEffect, useRef, useState, type FormEvent } from "react";

import { api } from "../api/client";
import type { SplitName, TaskType, VideoImportTask } from "../types";

type SamplingMode = "fps" | "interval_seconds" | "frame_interval";
type SplitStrategy = "single" | "time_blocks";

export function VideoImportModal({ onClose, onCompleted }: { onClose: () => void; onCompleted: (task: VideoImportTask) => Promise<void> | void }) {
  return <div className="modal-backdrop data-exchange-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="dataset-dialog data-exchange-dialog video-import-dialog" role="dialog" aria-modal="true" aria-labelledby="video-import-title" onMouseDown={(event) => event.stopPropagation()}>
      <header className="data-exchange-header"><div><span className="eyebrow">VIDEO IMPORT</span><h2 id="video-import-title">从视频创建数据集</h2><p>预检后再开始抽帧。连续帧很相似，自动随机 split 前请先检查数据泄漏风险。</p></div><button className="icon-button" onClick={onClose} aria-label="关闭">×</button></header>
      <VideoImportContent onClose={onClose} onCompleted={onCompleted} />
    </section>
  </div>;
}

export function VideoImportContent({ onClose, onCompleted }: { onClose: () => void; onCompleted: (task: VideoImportTask) => Promise<void> | void }) {
  const [file, setFile] = useState<File>();
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [split, setSplit] = useState<SplitName>("train");
  const [splitStrategy, setSplitStrategy] = useState<SplitStrategy>("single");
  const [timeBlockSeconds, setTimeBlockSeconds] = useState("30");
  const [trainRatio, setTrainRatio] = useState("0.8");
  const [valRatio, setValRatio] = useState("0.1");
  const [testRatio, setTestRatio] = useState("0.1");
  const [mode, setMode] = useState<SamplingMode>("fps");
  const [samplingValue, setSamplingValue] = useState("1");
  const [startSeconds, setStartSeconds] = useState("0");
  const [endSeconds, setEndSeconds] = useState("");
  const [task, setTask] = useState<VideoImportTask>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [draggingVideo, setDraggingVideo] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const isQueued = Boolean(task?.status === "pending" && task.start_requested);

  useEffect(() => {
    if (!task || (task.status !== "running" && !(task.status === "pending" && task.start_requested))) return;
    const interval = window.setInterval(() => {
      api.getVideoImport(task.id).then((next) => {
        setTask(next);
        if (next.status === "completed") void onCompleted(next);
      }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "无法读取导入进度。"));
    }, 1000);
    return () => window.clearInterval(interval);
  }, [task?.id, task?.status, task?.start_requested, onCompleted]);

  const selectFile = (next?: File) => {
    if (!next) return;
    setFile(next);
    if (!name) setName(next.name.replace(/\.[^.]+$/, ""));
  };
  const submitPreflight = async (event: FormEvent) => {
    event.preventDefault();
    if (!file || !name.trim()) return;
    setBusy(true);
    setError("");
    try {
      setTask(await api.createVideoImport({
        file,
        name: name.trim(),
        taskType,
        split,
        samplingMode: mode,
        samplingValue: Number(samplingValue),
        startSeconds: Number(startSeconds),
        endSeconds: endSeconds.trim() ? Number(endSeconds) : undefined,
        splitStrategy,
        timeBlockSeconds: Number(timeBlockSeconds),
        trainRatio: Number(trainRatio),
        valRatio: Number(valRatio),
        testRatio: Number(testRatio),
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "视频预检失败。");
    } finally {
      setBusy(false);
    }
  };
  const start = async () => {
    if (!task) return;
    setBusy(true);
    setError("");
    try { setTask(await api.startVideoImport(task.id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "无法开始导入。"); }
    finally { setBusy(false); }
  };
  const retry = async () => {
    if (!task) return;
    setBusy(true);
    setError("");
    try { setTask(await api.retryVideoImport(task.id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "无法重试导入。"); }
    finally { setBusy(false); }
  };
  const modeLabel = mode === "fps" ? "每秒抽帧数" : mode === "interval_seconds" ? "时间间隔（秒）" : "帧间隔";

  return <>{!task ? <form className="video-import-form" onSubmit={(event) => void submitPreflight(event)}>
        <input ref={inputRef} type="file" accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo" hidden onChange={(event) => { selectFile(event.target.files?.[0]); event.currentTarget.value = ""; }} />
        <button type="button" className={draggingVideo ? "exchange-dropzone dragging" : "exchange-dropzone"} onClick={() => inputRef.current?.click()} onDragEnter={(event) => { event.preventDefault(); setDraggingVideo(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDraggingVideo(false)} onDrop={(event) => { event.preventDefault(); setDraggingVideo(false); selectFile(event.dataTransfer.files[0]); }}><span className="exchange-dropzone-icon">⇧</span><strong>{file?.name ?? "拖放或选择 MP4、MOV 或 AVI 视频"}</strong><small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · 已准备预检` : "视频会先进入本机受管暂存目录"}</small></button>
        <div className="exchange-fields"><label>数据集名称<input value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：traffic-daytime" /></label><label>任务类型<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}><option value="detect">目标检测（Bounding Box）</option><option value="segment">实例分割（Polygon / SAM）</option><option value="obb">旋转框（OBB）</option><option value="classify">图像分类</option></select></label><label>split 策略<select value={splitStrategy} onChange={(event) => setSplitStrategy(event.target.value as SplitStrategy)}><option value="single">全部导入同一 split</option><option value="time_blocks">按连续时间块分配</option></select></label></div>
        {splitStrategy === "single" ? <div className="exchange-fields"><label>初始 split<select value={split} onChange={(event) => setSplit(event.target.value as SplitName)}><option value="train">train · 训练集</option><option value="val">val · 验证集</option><option value="test">test · 测试集</option></select></label></div> : <div className="exchange-fields video-sampling-fields"><label>时间块（秒）<input type="number" min="0.1" step="0.1" value={timeBlockSeconds} onChange={(event) => setTimeBlockSeconds(event.target.value)} /></label><label>train 比例<input type="number" min="0" max="1" step="0.05" value={trainRatio} onChange={(event) => setTrainRatio(event.target.value)} /></label><label>val 比例<input type="number" min="0" max="1" step="0.05" value={valRatio} onChange={(event) => setValRatio(event.target.value)} /></label><label>test 比例<input type="number" min="0" max="1" step="0.05" value={testRatio} onChange={(event) => setTestRatio(event.target.value)} /></label></div>}
        <div className="exchange-fields video-sampling-fields"><label>抽帧方式<select value={mode} onChange={(event) => setMode(event.target.value as SamplingMode)}><option value="fps">按 FPS</option><option value="interval_seconds">按时间间隔</option><option value="frame_interval">按帧间隔</option></select></label><label>{modeLabel}<input type="number" min={mode === "frame_interval" ? "1" : "0.001"} step={mode === "frame_interval" ? "1" : "any"} value={samplingValue} onChange={(event) => setSamplingValue(event.target.value)} /></label><label>开始（秒）<input type="number" min="0" step="0.1" value={startSeconds} onChange={(event) => setStartSeconds(event.target.value)} /></label><label>结束（秒，可选）<input type="number" min="0" step="0.1" value={endSeconds} onChange={(event) => setEndSeconds(event.target.value)} /></label></div>
        <p className="video-sampling-hint">{mode === "fps" ? "FPS=1 表示从所选开始时间起每秒生成 1 张；0.5 表示约每 2 秒 1 张。" : mode === "interval_seconds" ? "例如填 1 表示每隔 1 秒生成 1 张。" : "例如填 30 表示每隔 30 个源视频帧生成 1 张。"}</p>
        {error && <p className="video-import-error">{error}</p>}
        <footer className="data-exchange-footer"><span>核心版支持预检、进度、失败重试和帧来源记录。</span><div><button type="button" className="button" onClick={onClose}>取消</button><button className="button primary" disabled={busy || !file || !name.trim()}>预检视频</button></div></footer>
      </form> : <section className="video-import-result">
        <div className="exchange-summary"><span>分辨率<strong>{task.video_info_json.width} × {task.video_info_json.height}</strong></span><span>时长<strong>{task.video_info_json.duration_seconds.toFixed(1)} 秒</strong></span><span>预计帧数<strong>{task.total_images.toLocaleString()} 张</strong></span><span>预计输出<strong>{formatBytes(task.video_info_json.estimated_output_bytes)}</strong></span></div>
        {task.config_json.split_strategy === "time_blocks" && <p className="video-import-split-note">连续时间块分配：train {task.video_info_json.estimated_split_counts.train.toLocaleString()} / val {task.video_info_json.estimated_split_counts.val.toLocaleString()} / test {task.video_info_json.estimated_split_counts.test.toLocaleString()} 张（预估）。</p>}
        <div className="video-import-progress"><div><strong>{isQueued ? "正在排队" : task.status === "pending" ? "预检完成，等待开始" : task.status === "running" ? "正在抽帧" : task.status === "completed" ? "导入完成" : "导入失败"}</strong><span>{task.generated_images.toLocaleString()} / {task.total_images.toLocaleString()} 张 · {task.progress_percent.toFixed(0)}%</span></div><div className="progress-track"><i style={{ width: `${task.progress_percent}%` }} /></div></div>
        {task.status === "completed" && <p className="video-import-complete">数据集已创建。你现在可以关闭窗口，进入数据集检查图片与 split。</p>}
        {(error || task.error_message) && <p className="video-import-error">{error || task.error_message}</p>}
        <footer className="data-exchange-footer"><span>{isQueued ? "任务已确认，会在当前本地队列空闲后自动开始。" : task.status === "pending" ? "确认后才会开始生成图片。" : task.status === "running" ? "可以安全地留在此窗口查看进度。" : "视频源仍保留在受管暂存目录，失败时可重试。"}</span><div><button className="button" onClick={onClose}>{task.status === "running" || isQueued ? "后台继续" : "关闭"}</button>{task.status === "pending" && !isQueued && <button className="button primary" disabled={busy} onClick={() => void start()}>开始导入</button>}{task.status === "failed" && <button className="button primary" disabled={busy} onClick={() => void retry()}>重试导入</button>}</div></footer>
      </section>}</>;
}

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.ceil(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
