import type { AppLocale } from "../locale";

const sources = {
  zh: { llm: "LLM", mock: "本地规则（Mock）", fallback: "规则降级", mixed: "混合来源", unknown: "未记录" },
  en: { llm: "LLM", mock: "Local rules (Mock)", fallback: "Rule fallback", mixed: "Mixed sources", unknown: "Not recorded" },
};

const reasons: Record<string, [string, string]> = {
  agent_provider_http_error: ["LLM HTTP 请求失败", "LLM HTTP request failed"],
  agent_provider_unreachable: ["LLM 网络或超时错误", "LLM network error or timeout"],
  agent_provider_bad_response: ["LLM 响应格式无效", "Invalid LLM response"],
  agent_provider_unsupported: ["Provider 不受支持", "Unsupported provider"],
  no_tool_calls: ["模型未选择工具，改用规则规划", "Model selected no tools; rules planned the lookup"],
  missing_evaluation_lookup: ["模型未查询评估，规则补充查询", "Rules added the missing evaluation lookup"],
  provider_error: ["推理失败，详情见脱敏日志", "Inference failed; see redacted logs"],
};

export function inferenceSourceLabel(source: string | undefined, locale: AppLocale): string {
  return sources[locale][source as keyof typeof sources.zh] ?? sources[locale].unknown;
}

export function inferenceReasonLabel(reason: string, locale: AppLocale): string {
  return (reasons[reason] ?? reasons.provider_error)[locale === "zh" ? 0 : 1];
}
