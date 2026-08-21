import type { FormEvent } from "react";
import type { Dataset, TaskType } from "../../types";
import {
  BATCH_OPTIONS,
  IMG_SIZE_OPTIONS,
  PRESETS,
  modelChoicesFor,
  taskTypeLabel,
  type PresetId,
  type TrainingFormState,
} from "../../training/helpers";
import { IconBox, IconChevron, IconGrid, IconScan, IconShapes, IconSpark } from "./icons";

interface Props {
  datasets: Dataset[];
  dataset: Dataset;
  form: TrainingFormState;
  preset: PresetId | null;
  advancedOpen: boolean;
  summaryBottomBar: string;
  onDatasetChange: (dataset: Dataset) => void;
  onFormChange: (updater: (prev: TrainingFormState) => TrainingFormState) => void;
  onPreset: (preset: PresetId) => void;
  onAdvancedOpenChange: (open: boolean) => void;
  onSubmit: (event: FormEvent) => void;
}

const PRESET_COPY: Record<PresetId, { title: string; body: string; meta: string }> = {
  fast: { title: "快速验证", body: "适合冒烟与调通链路", meta: "30 ep · batch 16" },
  balanced: { title: "均衡训练", body: "日常默认推荐", meta: "50 ep · batch 16" },
  high: { title: "高精度", body: "更长收敛与更稳指标", meta: "100 ep · batch 8" },
};

const TASK_META: Record<TaskType, { help: string; Icon: typeof IconBox }> = {
  detect: { help: "绝对像素 bbox → YOLO detect 标签", Icon: IconBox },
  segment: { help: "polygon 标注；可选 SAM 建议后确认保存", Icon: IconShapes },
  obb: { help: "中心 / 尺寸 / 角度 → YOLO OBB", Icon: IconScan },
  classify: { help: "每图一类 → YOLO classify 目录布局", Icon: IconGrid },
};

export function TrainingConfigPanel({
  datasets,
  dataset,
  form,
  preset,
  advancedOpen,
  summaryBottomBar,
  onDatasetChange,
  onFormChange,
  onPreset,
  onAdvancedOpenChange,
  onSubmit,
}: Props) {
  const setForm = onFormChange;
  const modelChoices = modelChoicesFor(dataset.task_type);
  const modelInList = modelChoices.some((item) => item.value === form.model);
  const taskMeta = TASK_META[dataset.task_type];

  return (
    <section id="training-config-anchor" className="training-ws-card training-ws-config">
      <div className="training-ws-card-head">
        <div>
          <h2>训练配置</h2>
          <p className="training-ws-muted">选择权重与超参后提交，任务进入本机 FIFO 队列由 Ultralytics 执行。</p>
        </div>
      </div>

      <form id="dataset-training-ws-form" className="training-ws-form" onSubmit={onSubmit}>
        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>数据集与任务</h3>
            <p>任务类型由数据集锁定；切换数据集会重置表单默认值。</p>
          </div>
          <label className="training-ws-field training-ws-field-wide">
            <span>训练数据集</span>
            <select
              value={dataset.id}
              onChange={(event) => {
                const next = datasets.find((item) => item.id === event.target.value);
                if (next) onDatasetChange(next);
              }}
            >
              {datasets.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {taskTypeLabel(item.task_type)} · {item.image_count} 张
                </option>
              ))}
            </select>
          </label>
          <div className="tcc-task-type-seg tcc-task-type-seg--wrap" role="group" aria-label="任务类型">
            {(["detect", "segment", "obb", "classify"] as const).map((kind) => {
              const { Icon } = TASK_META[kind];
              return (
                <button
                  key={kind}
                  type="button"
                  className={`tcc-task-type-seg-btn${dataset.task_type === kind ? " active" : ""}`}
                  disabled={dataset.task_type !== kind}
                  aria-pressed={dataset.task_type === kind}
                >
                  <Icon size={15} />
                  <span>{taskTypeLabel(kind)}</span>
                </button>
              );
            })}
          </div>
          <p className="tcc-task-type-detail" role="status">{taskMeta.help}</p>
        </section>

        <section className="training-config-block">
          <div className="training-dataset-ai-card" role="region" aria-label="推荐配置">
            <div className="training-dataset-ai-head">
              <IconSpark size={16} />
              <span className="training-dataset-ai-badge">推荐起点</span>
              <span className="training-dataset-ai-active-pill">本地</span>
            </div>
            <div className="training-recommend-grid">
              <div>
                <small>任务</small>
                <strong>{taskTypeLabel(dataset.task_type)}</strong>
              </div>
              <div>
                <small>权重</small>
                <strong>{form.model || "yolo11n.pt"}</strong>
              </div>
              <div>
                <small>预设</small>
                <strong>均衡训练</strong>
              </div>
              <div>
                <small>数据</small>
                <strong>{dataset.image_count} 图 · {dataset.class_count} 类</strong>
              </div>
            </div>
            <p className="training-dataset-ai-reason">默认复用已保存的 train / val split；可随时覆盖下方参数。</p>
          </div>
        </section>

        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>训练预设</h3>
          </div>
          <div className="training-dataset-preset-cards" role="group" aria-label="训练预设">
            {(["fast", "balanced", "high"] as const).map((id) => (
              <button
                key={id}
                type="button"
                className={`training-dataset-preset-card${preset === id ? " active" : ""}`}
                onClick={() => onPreset(id)}
              >
                <strong>{PRESET_COPY[id].title}</strong>
                <span className="training-preset-meta">{PRESET_COPY[id].meta}</span>
                <span>{PRESET_COPY[id].body}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>模型与基础参数</h3>
          </div>
          <div className="training-ws-form-grid training-ws-form-grid--base">
            <label className="training-ws-field training-ws-field-wide">
              <span>实验名称</span>
              <input value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))} required />
            </label>
            <label className="training-ws-field training-ws-field-wide">
              <span>基础权重</span>
              <select
                value={modelInList ? form.model : "__custom__"}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === "__custom__") return;
                  setForm((prev) => ({ ...prev, model: value }));
                }}
              >
                {modelChoices.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
                {!modelInList && <option value="__custom__">{form.model}</option>}
              </select>
            </label>
            {!modelInList && (
              <label className="training-ws-field training-ws-field-wide">
                <span>自定义权重名</span>
                <input value={form.model} onChange={(event) => setForm((prev) => ({ ...prev, model: event.target.value }))} />
              </label>
            )}
            <label className="training-ws-field">
              <span>Epochs</span>
              <input type="number" min={1} value={form.epochs} onChange={(event) => setForm((prev) => ({ ...prev, epochs: Number(event.target.value) }))} />
            </label>
            <label className="training-ws-field">
              <span>Batch size</span>
              <select value={String(form.batch_size)} onChange={(event) => setForm((prev) => ({ ...prev, batch_size: Number(event.target.value) }))}>
                {!BATCH_OPTIONS.includes(form.batch_size) && <option value={form.batch_size}>{form.batch_size}</option>}
                {BATCH_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="training-ws-field">
              <span>Image size</span>
              <select value={String(form.img_size)} onChange={(event) => setForm((prev) => ({ ...prev, img_size: Number(event.target.value) }))}>
                {!IMG_SIZE_OPTIONS.includes(form.img_size) && <option value={form.img_size}>{form.img_size}px</option>}
                {IMG_SIZE_OPTIONS.map((n) => <option key={n} value={n}>{n}px</option>)}
              </select>
            </label>
          </div>
          <p className="tcc-model-hint training-ws-field-wide" role="note">
            首次使用 {form.model} 时，Ultralytics 可能下载对应权重。请确认适用许可后再训练。
          </p>
        </section>

        <div className="training-ws-advanced">
          <button type="button" className="training-ws-advanced-trigger" onClick={() => onAdvancedOpenChange(!advancedOpen)}>
            <span className={`training-ws-chevron${advancedOpen ? " flip" : ""}`} aria-hidden>
              <IconChevron size={16} />
            </span>
            <span className="training-ws-advanced-trigger-text">
              <strong>高级参数</strong>
              <span className="training-ws-muted-inline">优化器、学习率、patience、workers</span>
            </span>
          </button>
          {advancedOpen && (
            <div className="training-ws-advanced-grid">
              <label className="training-ws-field">
                <span>Val ratio（元数据）</span>
                <input type="number" step={0.05} min={0} max={0.9} value={form.val_ratio} onChange={(event) => setForm((prev) => ({ ...prev, val_ratio: Number(event.target.value) }))} />
              </label>
              <label className="training-ws-field">
                <span>Seed</span>
                <input type="number" value={form.seed} onChange={(event) => setForm((prev) => ({ ...prev, seed: Number(event.target.value) }))} />
              </label>
              <label className="training-ws-field">
                <span>学习率 lr0</span>
                <input value={form.lr0} placeholder="留空 = 默认" onChange={(event) => setForm((prev) => ({ ...prev, lr0: event.target.value }))} />
              </label>
              <label className="training-ws-field">
                <span>Optimizer</span>
                <input value={form.optimizer} placeholder="auto / AdamW / SGD" onChange={(event) => setForm((prev) => ({ ...prev, optimizer: event.target.value }))} />
              </label>
              <label className="training-ws-field">
                <span>Patience</span>
                <input value={form.patience} placeholder={String(PRESETS.balanced.patience)} onChange={(event) => setForm((prev) => ({ ...prev, patience: event.target.value }))} />
              </label>
              <label className="training-ws-field">
                <span>Workers</span>
                <input type="number" min={0} value={form.workers} onChange={(event) => setForm((prev) => ({ ...prev, workers: Number(event.target.value) }))} />
              </label>
            </div>
          )}
        </div>

        <div className="training-ws-meta-row">
          <span>{dataset.name}</span>
          <span>{dataset.image_count} 张 · {dataset.class_count} 类</span>
          <span>{taskTypeLabel(dataset.task_type)}</span>
        </div>

        <div className="training-config-summary-bar">
          <span>{summaryBottomBar}</span>
        </div>
      </form>
    </section>
  );
}
