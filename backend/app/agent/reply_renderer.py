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

    def format_tool_report(self, tool_results: list[dict[str, Any]], *, question: str = "") -> str:
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
        sections: list[str] = []
        problems: list[str] = []
        for item in tool_results:
            name = str(item.get("tool_name") or "tool")
            result = item.get("result") or {}
            if not isinstance(result, dict):
                sections.append(f"- {name}: {untrusted_text(result, limit=300)}")
                continue
            if result.get("error"):
                sections.append(f"- {name}: 失败 — {untrusted_text(result.get('error'), limit=300)}")
                problems.append(f"- {name} 查询失败，相关信息不足，不能视为对象或记录不存在。")
                continue
            sections.append(f"[{name}] {self._format_one(name, result)}")
            if result.get("found") is False:
                problems.append(f"- {name} 未找到可用记录，相关结论仍缺少证据。")
            if result.get("truncated") or result.get("issues_truncated"):
                problems.append(f"- {name} 结果有截断，不能据此排除未展示的问题。")
        next_steps: list[str] = []
        diagnostic_names = {item.get("tool_name") for item in tool_results}
        if "dataset_readiness" not in diagnostic_names and ("训练建议" in question or "training advice" in question.lower()):
            next_steps.extend(self._training_advice(tool_results))
        if not diagnostic_names.intersection({"model_comparability", "model_compare", "compare_models"}) and ("评估" in question or "evaluation" in question.lower()) and (
            "模型" in question or "model" in question.lower()
        ):
            problems.extend(self._evaluation_conclusion(tool_results))
        if "training_diagnose_failure" not in diagnostic_names and ("训练" in question or "training" in question.lower()) and any(
            word in question.lower() for word in ("失败", "为什么", "failed", "failure")
        ):
            names = {item.get("tool_name") for item in tool_results
                     if isinstance(item.get("result"), dict) and not item["result"].get("error")}
            if "training_logs" not in names:
                problems.append("- 尚未取得训练日志，无法确认失败根因；状态或报错摘要不能排除其他原因。")
            next_steps.append("- 结合失败状态与日志核实可能原因；调整参数后需人工确认才能提交新任务。")
        if not next_steps:
            next_steps.append("- 优先补齐上述缺失信息并核实对象与 split；任何写操作均需人工确认。")
        return "\n".join([
            "结论", intro,
            "证据", *sections,
            "问题", *(problems or ["- 当前结果仅覆盖已查询范围，不能据此推断未查询对象或整体质量。"]),
            "下一步", *next_steps,
        ])

    def _training_advice(self, results: list[dict[str, Any]]) -> list[str]:
        by_name = {str(item.get("tool_name")): item.get("result") or {} for item in results}
        quality = by_name.get("dataset_quality_report") or {}
        summary = quality.get("summary") or {}
        validation = by_name.get("dataset_validate") or by_name.get("validate_dataset") or {}
        validation_summary = validation.get("summary") or validation
        advice = ["训练建议（依据上面的本地数据，不会创建任务）："]
        errors = validation_summary.get("error_count")
        if isinstance(errors, int) and errors > 0:
            advice.append(f"- 先修复 {errors} 个校验错误，再考虑训练。")
        elif errors == 0:
            advice.append("- 当前校验未发现阻断错误；训练前仍需检查类别与 split 是否符合预期。")
        coverage = summary.get("coverage")
        if isinstance(coverage, (int, float)) and coverage < 1:
            advice.append(f"- 标注覆盖率为 {coverage * 100:.1f}%；优先核对未标注图片是否应进入训练。")
        elif not quality:
            advice.append("- 尚无质量报告，无法判断标注覆盖与类别分布。")
        tasks = (by_name.get("training_list") or by_name.get("list_training_tasks") or {}).get("items") or []
        failures = [row for row in tasks if row.get("status") == "failed"]
        if failures:
            advice.append(f"- 最近任务中有 {len(failures)} 个失败；先查看失败原因与日志，再调整参数。")
        if len(advice) == 1:
            advice.append("- 先检查数据集校验、类别分布与历史训练结果，再确定训练参数。")
        return advice

    def _evaluation_conclusion(self, results: list[dict[str, Any]]) -> list[str]:
        by_name = {str(item.get("tool_name")): item.get("result") or {} for item in results}
        evaluations = by_name.get("evaluation_list") or by_name.get("list_model_evaluations")
        if evaluations is None:
            return ["- 尚未读取评估记录，不能据此判断模型效果。"]
        if evaluations.get("error"):
            return ["- 评估记录查询失败，不能判断是否已有评估或模型效果。"]
        items = evaluations.get("items") or []
        if not items:
            return ["- 当前模型暂无评估记录；不能仅凭模型存在判断效果。"]
        latest = items[0]
        metrics = latest.get("metrics") or {}
        detail = f"指标 {_compact(metrics, limit=240)}" if metrics else "尚无可用指标"
        return [f"- 最近评估：{latest.get('status')}，split={latest.get('split')}；{detail}。"]

    def reply(self, tool_name: str, output: dict[str, Any]) -> str:
        text = self._format_one(tool_name, output)
        return text[2:] if text.startswith("- ") else text

    def _format_one(self, name: str, result: dict[str, Any]) -> str:
        if name in {"dataset_readiness", "training_diagnose_failure", "model_comparability"} or (name in {"model_compare", "compare_models"} and "comparable" in result):
            lines = [f"- {untrusted_text(result.get('summary'), limit=300)}"]
            for key in ("issues", "next_steps"):
                lines.extend(f"  · {untrusted_text(item, limit=300)}" for item in (result.get(key) or [])[:8])
            for item in (result.get("hypotheses") or [])[:5]:
                lines.append(f"  · 可能原因 {item.get('code')}：{untrusted_text(item.get('explanation'))}；证据 {item.get('evidence')}")
            return "\n".join(lines)
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
                    + (f" 失败原因={untrusted_text(row.get('error_message'), limit=160)}" if row.get('error_message') else "")
                )
            return "\n".join(lines)

        if name in {"get_training_task", "training_status"}:
            if result.get("found") is False:
                return "- 训练任务：未找到。"
            metrics = result.get("metrics") or {}
            return (
                f"- 训练任务详情：{untrusted_text(result.get('name'))} ({result.get('id')}) "
                f"status={result.get('status')} epoch={result.get('progress_epoch')}/"
                f"{result.get('progress_total_epochs')} metrics={_compact(metrics)}"
                + (f" 失败原因={untrusted_text(result.get('error_message'), limit=160)}" if result.get('error_message') else "")
                + "。"
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
                lines.append(
                    f"  · {row.get('id')} split={row.get('split')} status={row.get('status')}"
                    + (f" metrics={_compact(row.get('metrics'), limit=180)}" if row.get('metrics') else "")
                    + (f" error={untrusted_text(row.get('error_message'), limit=120)}" if row.get('error_message') else "")
                )
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
