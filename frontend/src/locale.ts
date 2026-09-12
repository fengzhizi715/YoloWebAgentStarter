export type AppLocale = "zh" | "en";

const STORAGE_KEY = "yolowebagent-starter-locale";

export function readLocale(): AppLocale {
  return window.localStorage.getItem(STORAGE_KEY) === "en" ? "en" : "zh";
}

export function saveLocale(locale: AppLocale): void {
  window.localStorage.setItem(STORAGE_KEY, locale);
  document.documentElement.lang = locale === "en" ? "en" : "zh-CN";
}

export const localeText = {
  zh: {
    datasets: "数据集",
    training: "训练",
    models: "模型",
    agent: "智能体",
    samSettings: "SAM 设置",
    languageSettings: "语言设置",
    logs: "日志",
    localMode: "本地模式",
    workspace: "我的工作区",
    community: "社区版",
    settingsGroup: "设置",
    workspaceGroup: "工作区",
    backToDatasets: "返回数据集",
    mainNav: "主导航",
    currentDataset: "当前数据集：",
    sidebarNoteTitle: "本地 YOLO 工作台",
    sidebarNoteBody: "数据和模型仅保存在此设备。",
  },
  en: {
    datasets: "Datasets",
    training: "Training",
    models: "Models",
    agent: "Agent",
    samSettings: "SAM settings",
    languageSettings: "Language",
    logs: "Logs",
    localMode: "Local mode",
    workspace: "Workspace",
    community: "Community",
    settingsGroup: "Settings",
    workspaceGroup: "Workspace",
    backToDatasets: "Back to datasets",
    mainNav: "Main navigation",
    currentDataset: "Current dataset: ",
    sidebarNoteTitle: "Local YOLO workspace",
    sidebarNoteBody: "Data and models stay on this device.",
  },
} as const;
