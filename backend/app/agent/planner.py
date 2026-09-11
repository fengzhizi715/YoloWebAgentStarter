"""Community RuleBasedPlanner — trimmed from upstream without Workflow/AL/Deployment/Inspection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.provider import ProviderToolCall
from app.agent.write_tools import WRITE_TOOL_NAMES


_ID_RE = re.compile(r"\b((?:ds|train|eval)_[a-f0-9]+|model_[a-zA-Z0-9_]+)\b", re.IGNORECASE)

# Upstream dotted names → Starter OpenAI-safe names.
_PRIMARY_ALIASES: dict[str, tuple[str, ...]] = {
    "global_summary": ("list_datasets",),
    "dataset_summary": ("get_dataset",),
    "dataset_quality_report": (),
    "dataset_validate": ("validate_dataset",),
    "training_list": ("list_training_tasks",),
    "training_status": ("get_training_task",),
    "training_latest_result": ("training_summary",),
    "training_logs": (),
    "model_list": ("list_models",),
    "model_get": ("get_model",),
    "model_latest_for_dataset": (),
    "model_compare": ("compare_models",),
    "evaluation_list": ("list_model_evaluations",),
    "evaluation_summary": ("get_model_evaluation",),
    "evaluation_error_summary": (),
    "create_training_task": (),
    "create_model_evaluation": (),
    "create_auto_annotation_task": (),
}


@dataclass
class AgentPlanStep:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    requires_confirmation: bool = False


@dataclass
class AgentStructuredPlan:
    intent: str
    steps: list[AgentPlanStep] = field(default_factory=list)
    needs_confirmation: bool = False
    risks: list[str] = field(default_factory=list)


class RuleBasedPlanner:
    """Keyword planner for mock / fallback. Only Community-scoped tools."""

    def structured_plan(self, message: str, *, available: set[str] | None = None) -> AgentStructuredPlan:
        text = (message or "").lower()
        ids = _ID_RE.findall(message or "")
        dataset_ids = [item for item in ids if item.lower().startswith("ds_")]
        train_ids = [item for item in ids if item.lower().startswith("train_")]
        model_ids = [item for item in ids if item.lower().startswith("model_")]
        eval_ids = [item for item in ids if item.lower().startswith("eval_")]
        allow = available

        # Reject out-of-scope automation keywords; still offer a safe read tool.
        if self._mentions_out_of_scope(text):
            step = self._step("global_summary", {}, allow)
            return AgentStructuredPlan(
                intent="out_of_scope_hint",
                steps=[step] if step else [],
                needs_confirmation=False,
                risks=[
                    "Community Agent 不支持 Workflow、主动学习、Deployment、Inspection 或无人值守自动串联。"
                ],
            )

        wants_write_training = self._wants_training_write(text)
        wants_write_eval = any(
            token in text
            for token in ("创建评估", "开始评估", "发起评估", "跑评估", "create evaluation", "start evaluation", "run evaluation")
        )
        wants_write_auto = any(
            token in text
            for token in ("自动标注", "auto-annotation", "auto annotation", "创建自动标注", "start auto")
        )

        if wants_write_auto and dataset_ids and model_ids:
            step = self._step(
                "create_auto_annotation_task",
                {
                    "dataset_id": dataset_ids[0],
                    "model_id": model_ids[0],
                    "confidence": self._number(text, "confidence", 0.25),
                    "iou": self._number(text, "iou", 0.45),
                },
                allow,
                requires_confirmation=True,
            )
            return self._single(step, risks=self._risks("create_auto_annotation_task"))

        if wants_write_eval and model_ids:
            step = self._step(
                "create_model_evaluation",
                {
                    "model_id": model_ids[0],
                    "split": "val",
                    "confidence": self._number(text, "confidence", 0.25),
                    "iou": self._number(text, "iou", 0.5),
                },
                allow,
                requires_confirmation=True,
            )
            return self._single(step, risks=self._risks("create_model_evaluation"))

        # validate → propose training (no export / no auto-eval chaining)
        if (
            dataset_ids
            and any(word in text for word in ("训练", "epoch", "yolo"))
            and (
                wants_write_training
                or (any(word in text for word in ("如果没问题", "然后", "再", "先", "校验后")) and "训练" in text)
            )
        ):
            steps: list[AgentPlanStep] = []
            if any(word in text for word in ("如果没问题", "然后", "再", "先", "校验", "validate", "检查")):
                validate = self._step("dataset_validate", {"dataset_id": dataset_ids[0]}, allow)
                if validate:
                    steps.append(validate)
            train = self._step(
                "create_training_task",
                {
                    "dataset_id": dataset_ids[0],
                    "name": "agent-training",
                    "model": self._model(text),
                    "epochs": self._epochs(text),
                    "img_size": int(self._number(text, "img", 640)),
                    "batch_size": int(self._number(text, "batch", 16)),
                    "device": "auto",
                    "val_ratio": self._number(text, "val", 0.2),
                    "seed": int(self._number(text, "seed", 42)),
                },
                allow,
                requires_confirmation=True,
            )
            if train:
                steps.append(train)
            if steps:
                risks = [
                    "只读校验会自动执行；训练任务需你在确认卡中单独批准。",
                    *self._risks("create_training_task"),
                ]
                return AgentStructuredPlan(
                    intent="validate_then_train" if len(steps) > 1 else "create_training_task",
                    steps=steps,
                    needs_confirmation=True,
                    risks=risks,
                )

        if wants_write_training and dataset_ids:
            step = self._step(
                "create_training_task",
                {
                    "dataset_id": dataset_ids[0],
                    "name": "agent-training",
                    "model": self._model(text),
                    "epochs": self._epochs(text),
                    "img_size": int(self._number(text, "img", 640)),
                    "batch_size": int(self._number(text, "batch", 16)),
                    "device": "auto",
                    "val_ratio": self._number(text, "val", 0.2),
                    "seed": int(self._number(text, "seed", 42)),
                },
                allow,
                requires_confirmation=True,
            )
            return self._single(step, risks=self._risks("create_training_task"))

        wants_quality = any(token in text for token in ("quality", "质量", "分布", "coverage", "覆盖"))
        wants_validate = any(token in text for token in ("validate", "校验", "验证"))
        wants_training = any(token in text for token in ("train", "训练", "epoch", "loss"))
        wants_logs = any(token in text for token in ("log", "日志"))
        wants_models = any(token in text for token in ("model", "模型", "map", "权重"))
        wants_compare = any(token in text for token in ("compare", "对比", "比较"))
        wants_eval = any(token in text for token in ("eval", "评估", "validation", "val "))
        wants_dataset = any(token in text for token in ("dataset", "数据集", "概览", "overview"))
        wants_errors = any(token in text for token in ("error", "错误样本", "误检", "漏检"))

        if wants_compare and len(model_ids) >= 2:
            step = self._step(
                "model_compare",
                {"baseline_model_id": model_ids[0], "candidate_model_id": model_ids[1]},
                allow,
            )
            return self._single(step)

        if wants_eval and model_ids:
            if wants_errors:
                step = self._step("evaluation_error_summary", {"model_id": model_ids[0]}, allow)
                return self._single(step)
            if eval_ids:
                step = self._step(
                    "evaluation_summary",
                    {"model_id": model_ids[0], "evaluation_id": eval_ids[0]},
                    allow,
                )
            else:
                step = self._step("evaluation_list", {"model_id": model_ids[0]}, allow)
            return self._single(step)

        if wants_models and model_ids:
            step = self._step("model_get", {"model_id": model_ids[0]}, allow)
            return self._single(step)

        if wants_models and dataset_ids:
            step = self._step("model_latest_for_dataset", {"dataset_id": dataset_ids[0]}, allow) or self._step(
                "model_list", {"dataset_id": dataset_ids[0]}, allow
            )
            return self._single(step)

        if wants_models and not dataset_ids and not train_ids:
            step = self._step("model_list", {}, allow)
            return self._single(step)

        if wants_training and train_ids:
            if wants_logs:
                step = self._step("training_logs", {"task_id": train_ids[0], "tail": 40}, allow)
            else:
                step = self._step("training_latest_result", {"task_id": train_ids[0]}, allow) or self._step(
                    "training_status", {"task_id": train_ids[0]}, allow
                )
            return self._single(step)

        if wants_training and dataset_ids:
            step = self._step("training_list", {"dataset_id": dataset_ids[0]}, allow)
            return self._single(step)

        if wants_training:
            step = self._step("training_list", {}, allow)
            return self._single(step)

        if dataset_ids:
            dataset_id = dataset_ids[0]
            steps = []
            if wants_quality:
                call = self._step("dataset_quality_report", {"dataset_id": dataset_id}, allow)
                if call:
                    steps.append(call)
            if wants_validate:
                call = self._step("dataset_validate", {"dataset_id": dataset_id}, allow)
                if call:
                    steps.append(call)
            if not steps:
                call = self._step("dataset_summary", {"dataset_id": dataset_id}, allow)
                if call:
                    steps.append(call)
                if wants_dataset or wants_quality:
                    quality = self._step("dataset_quality_report", {"dataset_id": dataset_id}, allow)
                    if quality and all(item.tool_name != quality.tool_name for item in steps):
                        steps.append(quality)
            return AgentStructuredPlan(
                intent=steps[0].tool_name if steps else "noop",
                steps=steps,
                needs_confirmation=False,
                risks=[],
            )

        if wants_quality or wants_validate or wants_dataset or not text.strip():
            step = self._step("global_summary", {}, allow)
            return self._single(step)

        if any(token in text for token in ("什么", "哪些", "how", "what", "list", "列表")):
            step = self._step("global_summary", {}, allow)
            return self._single(step)

        return AgentStructuredPlan(intent="noop", steps=[], needs_confirmation=False, risks=[])

    def plan_tool_calls(self, message: str, available: set[str]) -> list[ProviderToolCall]:
        plan = self.structured_plan(message, available=available)
        calls: list[ProviderToolCall] = []
        for step in plan.steps:
            calls.append(
                ProviderToolCall(
                    name=step.tool_name,
                    arguments=step.arguments,
                    call_id=f"mock_{step.tool_name}",
                )
            )
        return calls

    def _single(self, step: AgentPlanStep | None, *, risks: list[str] | None = None) -> AgentStructuredPlan:
        if step is None:
            return AgentStructuredPlan(intent="noop", steps=[], needs_confirmation=False, risks=[])
        return AgentStructuredPlan(
            intent=step.tool_name,
            steps=[step],
            needs_confirmation=step.requires_confirmation,
            risks=risks or ([] if not step.requires_confirmation else self._risks(step.tool_name)),
        )

    def _step(
        self,
        primary: str,
        arguments: dict[str, Any],
        available: set[str] | None,
        *,
        requires_confirmation: bool | None = None,
    ) -> AgentPlanStep | None:
        name = self._resolve(primary, available)
        if name is None:
            return None
        confirm = bool(requires_confirmation) if requires_confirmation is not None else name in WRITE_TOOL_NAMES
        return AgentPlanStep(
            tool_name=name,
            arguments=arguments,
            description=self._description(primary),
            requires_confirmation=confirm,
        )

    def _resolve(self, primary: str, available: set[str] | None) -> str | None:
        candidates = (primary, *_PRIMARY_ALIASES.get(primary, ()))
        if available is None:
            return primary
        for candidate in candidates:
            if candidate in available:
                return candidate
        return None

    @staticmethod
    def _wants_training_read(text: str) -> bool:
        return any(
            token in text
            for token in (
                "训练状态",
                "训练进度",
                "训练日志",
                "训练结果",
                "训练摘要",
                "训练列表",
                "训练情况",
                "训练得",
                "查看训练",
                "列出训练",
                "怎么样",
                "如何",
                "progress",
                "status",
                "logs",
                "latest",
                "loss",
            )
        )

    @classmethod
    def _wants_training_write(cls, text: str) -> bool:
        explicit = (
            "创建训练",
            "开始训练",
            "发起训练",
            "提交训练",
            "create training",
            "start training",
            "submit training",
        )
        if any(token in text for token in explicit):
            return True
        if cls._wants_training_read(text):
            return False
        if re.search(r"\d+\s*(?:个\s*)?epoch", text):
            return True
        if re.search(r"(yolo[\w.-]*\.pt|best\.pt|last\.pt)", text) and "训练" in text:
            return True
        return False

    @staticmethod
    def _mentions_out_of_scope(text: str) -> bool:
        keywords = (
            "workflow",
            "主动学习",
            "active learning",
            "deployment",
            "边缘部署",
            "inspection",
            "工业质检",
            "memory bank",
            "定时",
            "触发器",
            "无人值守",
            "自动串联",
        )
        return any(word in text for word in keywords)

    def _description(self, tool_name: str) -> str:
        return {
            "global_summary": "汇总本地数据集、训练与模型概况",
            "dataset_summary": "查看数据集概览",
            "dataset_quality_report": "生成数据集质量报告",
            "dataset_validate": "校验数据集是否存在阻塞训练的问题",
            "training_list": "列出训练任务",
            "training_status": "查看训练任务状态",
            "training_latest_result": "查看最近一次训练结果",
            "training_logs": "查看训练日志尾部",
            "model_list": "列出受管模型",
            "model_get": "查看受管模型详情",
            "model_latest_for_dataset": "查看数据集最近的受管 PT",
            "model_compare": "对比两个受管模型指标",
            "evaluation_list": "列出模型评估记录",
            "evaluation_summary": "查看评估摘要",
            "evaluation_error_summary": "汇总评估错误样本类型",
            "create_training_task": "创建本地 YOLO 训练任务（需确认）",
            "create_model_evaluation": "创建受管模型评估任务（需确认）",
            "create_auto_annotation_task": "创建自动标注任务（需确认，结果须人工审核）",
        }.get(tool_name, tool_name)

    def _risks(self, tool_name: str) -> list[str]:
        if tool_name == "create_training_task":
            return ["训练会创建后台任务并占用本机计算资源。确认后本轮结束，不会自动启动评估。"]
        if tool_name == "create_model_evaluation":
            return ["评估会创建后台任务并读取已持久化 split。"]
        if tool_name == "create_auto_annotation_task":
            return ["自动标注会写入 source=auto 的标注，必须人工审核。"]
        return []

    def _epochs(self, text: str) -> int:
        match = re.search(r"(\d+)\s*(?:个\s*)?epoch", text)
        return int(match.group(1)) if match else 50

    def _model(self, text: str, default: str = "yolo11n.pt") -> str:
        match = re.search(r"(yolo[\w.-]*\.pt|best\.pt|last\.pt)", text)
        return match.group(1) if match else default

    def _number(self, text: str, key: str, default: float) -> float:
        match = re.search(rf"{key}\s*[=:：]?\s*(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else default
