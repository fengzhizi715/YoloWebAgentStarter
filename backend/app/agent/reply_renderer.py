"""Community reply renderer — trimmed from upstream for Starter tool names."""

from __future__ import annotations

import json
from typing import Any

from app.agent.bounds import bound_log_text, untrusted_text
from app.agent.planner import AgentStructuredPlan


class AgentReplyRenderer:
    def confirmation_reply(self, plan: AgentStructuredPlan) -> str:
        if not plan.steps:
            return "已生成待确认计划，请审核参数后再执行。"
        step_text = "；".join(
            f"{index + 1}. {step.description or step.tool_name}" for index, step in enumerate(plan.steps)
        )
        risks = f" 风险：{'；'.join(plan.risks)}" if plan.risks else ""
        return f"我已生成执行计划，需要你确认后继续：{step_text}。{risks} 确认后本轮结束，不会自动串联下一步。"

    def multi_reply(self, outputs: list[tuple[str, dict[str, Any]]]) -> str:
        if not outputs:
            return "没有可用的工具结果。"
        if len(outputs) == 1:
            return self.reply(outputs[0][0], outputs[0][1])
        return "已完成多步查询：" + "；".join(self.reply(tool_name, output) for tool_name, output in outputs)

    def format_tool_report(self, tool_results: list[dict[str, Any]]) -> str:
        if not tool_results:
            return "没有可用的工具结果。"
        has_write = any(
            isinstance(item.get("result"), dict) and (item.get("result") or {}).get("requires_approval")
            for item in tool_results
        )
        if has_write:
            intro = "以下为待人工确认的写操作预览（确认前不会创建任务）。"
        else:
            intro = "以下结论均来自本地只读工具结果（未发送原始图片或完整日志）。"
        sections: list[str] = [intro]
        for item in tool_results:
            name = str(item.get("tool_name") or "tool")
            result = item.get("result") or {}
            if not isinstance(result, dict):
                sections.append(f"- {name}: {untrusted_text(result, limit=300)}")
                continue
            if result.get("error"):
                sections.append(f"- {name}: 失败 — {untrusted_text(result.get('error'), limit=300)}")
                continue
            sections.append(self._format_one(name, result))
        return "\n".join(sections)

    def reply(self, tool_name: str, output: dict[str, Any]) -> str:
        text = self._format_one(tool_name, output)
        return text[2:] if text.startswith("- ") else text

    def _format_one(self, name: str, result: dict[str, Any]) -> str:
        if result.get("requires_approval"):
            preview = result.get("preview") or result
            return f"- 待确认：{name} preview={_compact(preview)}。"

        if name in {"list_datasets", "global_summary"}:
            items = result.get("items") or []
            if "dataset_count" in result:
                return (
                    f"- 全局概况：数据集 {result.get('dataset_count')}，"
                    f"训练 {result.get('training_task_count')}，模型 {result.get('model_count')}。"
                )
            lines = [f"- 数据集列表：共 {result.get('total', len(items))} 个（展示 {len(items)} 个）。"]
            for row in items[:10]:
                lines.append(
                    f"  · {untrusted_text(row.get('name'))} ({row.get('id')}) "
                    f"task={row.get('task_type')} images={row.get('image_count')} "
                    f"annotated={row.get('annotated_image_count')} classes={row.get('class_count')}"
                )
            return "\n".join(lines)

        if name in {"get_dataset", "dataset_summary"}:
            return (
                f"- 数据集概览：{untrusted_text(result.get('name'))} ({result.get('id')})，"
                f"任务 {result.get('task_type')}，图片 {result.get('image_count')}，"
                f"已标注 {result.get('annotated_image_count')}，类别 {result.get('class_count')}。"
                f" split={result.get('split_counts')}。"
            )

        if name == "dataset_quality_report":
            summary = result.get("summary") or {}
            dist = result.get("class_distribution") or []
            issues = result.get("issues") or []
            top = ", ".join(f"{untrusted_text(item.get('name'))}:{item.get('count')}" for item in dist[:8]) or "无"
            coverage = summary.get("coverage")
            if isinstance(coverage, (int, float)) and coverage <= 1:
                coverage_text = f"{coverage * 100:.1f}%"
            else:
                coverage_text = str(coverage)
            return (
                f"- 质量报告：覆盖率 {coverage_text}，标注 {summary.get('annotation_count')}，"
                f"小目标占比 {summary.get('small_object_ratio')}；类别分布 [{top}]；"
                f"问题 {len(issues)} 条（另有截断 {result.get('issues_truncated', 0)}）。"
            )

        if name in {"validate_dataset", "dataset_validate"}:
            if "summary" in result and isinstance(result["summary"], dict):
                summary = result["summary"]
                return (
                    f"- 校验：{result.get('status') or result.get('valid')}，"
                    f"错误 {summary.get('error_count')}，警告 {summary.get('warning_count')}。"
                )
            return (
                f"- 校验：valid={result.get('valid')}，"
                f"错误 {result.get('error_count')}，警告 {result.get('warning_count')}；"
                f"展示问题 {len(result.get('issues') or [])} 条。"
            )

        if name in {"list_training_tasks", "training_list"}:
            items = result.get("items") or []
            lines = [f"- 训练任务：共 {result.get('total', len(items))} 个（展示 {len(items)} 个）。"]
            for row in items[:10]:
                lines.append(
                    f"  · {untrusted_text(row.get('name'))} ({row.get('id')}) "
                    f"status={row.get('status')} progress={row.get('progress_percent')}%"
                )
            return "\n".join(lines)

        if name in {"get_training_task", "training_status"}:
            if result.get("found") is False:
                return "- 训练任务：未找到。"
            metrics = result.get("metrics") or {}
            return (
                f"- 训练任务详情：{untrusted_text(result.get('name'))} ({result.get('id')}) "
                f"status={result.get('status')} epoch={result.get('progress_epoch')}/"
                f"{result.get('progress_total_epochs')} metrics={_compact(metrics)}。"
            )

        if name in {"training_summary", "training_latest_result"}:
            if result.get("found") is False:
                return "- 还没有训练任务。"
            progress = result.get("progress") or {}
            metrics = result.get("metrics") or result.get("latest_metrics") or {}
            risks = result.get("risks") or []
            return (
                f"- 训练摘要：status={result.get('status')} progress={_compact(progress)} "
                f"metrics={_compact(metrics)} risks={len(risks)}。"
            )

        if name == "training_logs":
            if result.get("found") is False:
                return "- 还没有训练日志。"
            preview = bound_log_text(result.get("preview") or result.get("logs") or "", max_chars=400, max_lines=8)
            return f"- 训练日志摘要：总行数 {result.get('line_count')}，预览：{preview or '(empty)'}。"

        if name in {"list_models", "model_list"}:
            items = result.get("items") or []
            lines = [f"- 模型列表：共 {result.get('total', len(items))} 个（展示 {len(items)} 个）。"]
            for row in items[:10]:
                lines.append(
                    f"  · {untrusted_text(row.get('name'))} ({row.get('id')}) "
                    f"mAP50={row.get('map50')} status={row.get('status')}"
                )
            return "\n".join(lines)

        if name in {"get_model", "model_get"}:
            return (
                f"- 模型详情：{untrusted_text(result.get('name'))} ({result.get('id')}) "
                f"task={result.get('task_type')} P/R/mAP50/mAP50-95="
                f"{result.get('precision')}/{result.get('recall')}/{result.get('map50')}/{result.get('map50_95')}；"
                f"备注={untrusted_text(result.get('notes'), limit=120)}。"
            )

        if name == "model_latest_for_dataset":
            if result.get("found") is False:
                return "- 模型：该数据集暂无受管 PT。"
            model = result.get("model") or {}
            return (
                f"- 最新受管模型：{untrusted_text(model.get('name'))} ({model.get('id')}) "
                f"map50={model.get('map50')}。"
            )

        if name in {"compare_models", "model_compare"}:
            if result.get("found") is False:
                return "- 当前还没有足够的模型版本可比较。"
            deltas = result.get("deltas") or {}
            suggestions = result.get("suggestions") or []
            return (
                f"- 模型对比：baseline={result.get('baseline', {}).get('id')} "
                f"candidate={result.get('candidate', {}).get('id')} deltas={_compact(deltas)}；"
                f"建议：{untrusted_text('; '.join(str(s) for s in suggestions[:3]), limit=240)}。"
            )

        if name in {"list_model_evaluations", "evaluation_list"}:
            items = result.get("items") or []
            lines = [f"- 评估记录：共 {result.get('total', len(items))} 个（展示 {len(items)} 个）。"]
            for row in items[:10]:
                lines.append(f"  · {row.get('id')} split={row.get('split')} status={row.get('status')}")
            return "\n".join(lines)

        if name in {"get_model_evaluation", "evaluation_summary"}:
            if result.get("found") is False:
                return "- 还没有评估任务。"
            metrics = result.get("metrics") or {}
            return (
                f"- 评估详情：{result.get('id')} split={result.get('split')} "
                f"status={result.get('status')} metrics={_compact(metrics)}。"
            )

        if name == "evaluation_error_summary":
            if result.get("found") is False:
                return "- 评估错误摘要：暂无评估记录。"
            return (
                f"- 评估错误摘要：sample_count={result.get('sample_count')} "
                f"types={_compact(result.get('error_type_counts') or result.get('error_type_breakdown') or [])}。"
            )

        if name == "create_training_task":
            preview = result.get("preview") or result
            return (
                f"- 待确认：创建训练任务 dataset={preview.get('dataset_id')} "
                f"name={untrusted_text(preview.get('name'))} model={untrusted_text(preview.get('model'))} "
                f"epochs={preview.get('epochs')}。"
            )
        if name == "create_model_evaluation":
            preview = result.get("preview") or result
            return (
                f"- 待确认：创建模型评估 model={preview.get('model_id')} "
                f"split={preview.get('split')} conf={preview.get('confidence')} iou={preview.get('iou')}。"
            )
        if name == "create_auto_annotation_task":
            preview = result.get("preview") or result
            return (
                f"- 待确认：创建自动标注 dataset={preview.get('dataset_id')} "
                f"model={preview.get('model_id')}。"
            )

        return f"- {name}: {untrusted_text(json.dumps(result, ensure_ascii=False, default=str), limit=400)}"


def _compact(value: Any, *, limit: int = 180) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return untrusted_text(text, limit=limit)
