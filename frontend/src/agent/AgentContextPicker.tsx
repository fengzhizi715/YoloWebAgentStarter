import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AgentContext, AgentProfileId, Dataset, TrainingTask, ModelVersion } from "../types";
import type { AppLocale } from "../locale";
import { PROFILE_IDS, profileTitle } from "./profiles";
import { IconBox, IconCpu, IconGrid, IconSpark } from "../components/training/icons";

const profileMeta = {
  global: { Icon: IconSpark, zh: "跨领域问答与工作区概览", en: "Workspace overview & questions" },
  dataset: { Icon: IconGrid, zh: "数据质量与训练准备度", en: "Data quality & training readiness" },
  training: { Icon: IconCpu, zh: "训练进度与失败诊断", en: "Progress & failure diagnosis" },
  model: { Icon: IconBox, zh: "评估结果与版本比较", en: "Evaluation & version comparison" },
};

export function AgentContextPicker({ locale, profile, context, disabled, onChange }: {
  locale: AppLocale; profile: AgentProfileId; context: AgentContext; disabled: boolean;
  onChange: (profile: AgentProfileId, context: AgentContext) => void;
}) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    let disposed = false;
    setError(""); setTasks([]); setModels([]);
    void Promise.all([api.listDatasets(), api.listTrainingTasks(context.dataset_id ?? undefined), api.listModels(context.dataset_id ?? undefined)])
      .then(([ds, ts, ms]) => { if (!disposed) { setDatasets(ds); setTasks(ts.items); setModels(ms.items); } })
      .catch((reason) => { if (!disposed) setError(reason instanceof Error ? reason.message : "Context lookup failed"); });
    return () => { disposed = true; };
  }, [context.dataset_id]);
  const none = locale === "zh" ? "未选择" : "Not selected";
  return <section className="agent-binding-panel" aria-label={locale === "zh" ? "助手模式与对象" : "Assistant mode and context"}>
    <div className="agent-profile-options">{PROFILE_IDS.map((id) => { const { Icon } = profileMeta[id]; return <button type="button" className="button" key={id}
      aria-label={profileTitle(id, locale)}
      aria-pressed={profile === id} disabled={disabled} onClick={() => onChange(id,
        id === "global" ? {} : id === "dataset" ? { dataset_id: context.dataset_id } : id === "training"
          ? { dataset_id: context.dataset_id, training_task_id: context.training_task_id }
          : { dataset_id: context.dataset_id, model_id: context.model_id })}>
        <span className="agent-profile-icon"><Icon size={19} /></span>
        <span><strong>{profileTitle(id, locale)}</strong><small>{profileMeta[id][locale]}</small></span>
      </button>; })}</div>
    <div className="agent-context-selectors">
      <label>{locale === "zh" ? "数据集" : "Dataset"}<select disabled={disabled} value={context.dataset_id ?? ""}
        onChange={(event) => onChange(profile === "global" && event.target.value ? "dataset" : profile, event.target.value ? { dataset_id: event.target.value } : {})}>
        <option value="">{none}</option>
        {context.dataset_id && !datasets.some((d) => d.id === context.dataset_id) ? <option value={context.dataset_id}>{context.dataset_id}</option> : null}
        {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select></label>
      {profile === "training" || context.training_task_id ? <label>{locale === "zh" ? "训练任务" : "Training task"}<select disabled={disabled} value={context.training_task_id ?? ""}
        onChange={(event) => { const task = tasks.find((item) => item.id === event.target.value); onChange("training", { dataset_id: task?.dataset_id ?? context.dataset_id, training_task_id: task?.id }); }}>
        <option value="">{none}</option>
        {context.training_task_id && !tasks.some((t) => t.id === context.training_task_id) ? <option value={context.training_task_id}>{context.training_task_id}</option> : null}
        {tasks.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
      </select></label> : null}
      {profile === "model" || context.model_id ? <label>{locale === "zh" ? "模型" : "Model"}<select disabled={disabled} value={context.model_id ?? ""}
        onChange={(event) => { const model = models.find((item) => item.id === event.target.value); onChange("model", { dataset_id: model?.dataset_id ?? context.dataset_id, model_id: model?.id }); }}>
        <option value="">{none}</option>
        {context.model_id && !models.some((m) => m.id === context.model_id) ? <option value={context.model_id}>{context.model_id}</option> : null}
        {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select></label> : null}
      <span className="agent-context-hint">{locale === "zh" ? "选择对象后，提问会自动携带上下文" : "Selected objects provide context for your questions"}</span>
    </div>
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
