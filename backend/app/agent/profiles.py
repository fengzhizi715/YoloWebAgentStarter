"""Versioned, built-in scene policies; not independent agents or workflows."""

from dataclasses import dataclass
from typing import Literal

from app.core.errors import ValidationError

ProfileId = Literal["global", "dataset", "training", "model"]


@dataclass(frozen=True)
class AgentProfile:
    id: ProfileId
    title: str
    title_en: str
    instructions: str
    allowed_tools: frozenset[str]
    version: int = 1


DATA_READ = {"global_summary", "dataset_summary", "dataset_quality_report", "dataset_validate", "dataset_readiness"}
TRAIN_READ = {"training_list", "training_status", "training_latest_result", "training_logs", "training_diagnose_failure"}
MODEL_READ = {"model_list", "model_get", "model_latest_for_dataset", "model_compare", "model_comparability",
              "evaluation_list", "evaluation_summary", "evaluation_error_summary"}

PROFILES = {
    "global": AgentProfile("global", "综合助手", "General assistant",
        "定位对象、回答跨领域问题。没有明确对象时先列出候选供用户选择，不要擅自选择写操作目标。",
        frozenset(DATA_READ | TRAIN_READ | MODEL_READ | {"create_training_task", "create_model_evaluation", "create_auto_annotation_task"})),
    "dataset": AgentProfile("dataset", "数据集助手", "Dataset assistant",
        "重点分析数据质量与训练准备度。训练准备报告优先使用 dataset_readiness；可跨领域读取训练和模型证据。",
        frozenset(DATA_READ | TRAIN_READ | MODEL_READ | {"create_training_task", "create_auto_annotation_task"})),
    "training": AgentProfile("training", "训练助手", "Training assistant",
        "重点解释训练进度和失败原因。失败诊断优先使用 training_diagnose_failure；相关数据校验可只读查询。"
        "参数修改只是建议，重提训练或评估仍须人工确认。",
        frozenset(DATA_READ | TRAIN_READ | MODEL_READ | {"create_training_task", "create_model_evaluation"})),
    "model": AgentProfile("model", "模型助手", "Model assistant",
        "重点解释评估、错误样本和版本差异。比较前优先使用 model_comparability；可比性未知时不推荐直接替换模型。",
        frozenset(DATA_READ | MODEL_READ | {"training_status", "training_latest_result", "create_model_evaluation", "create_auto_annotation_task"})),
}


def infer_profile(context: dict) -> ProfileId:
    return "model" if context.get("model_id") else "training" if context.get("training_task_id") else "dataset" if context.get("dataset_id") else "global"


def get_profile(profile_id: str, version: int = 1) -> AgentProfile:
    profile = PROFILES.get(profile_id)
    if profile is None or profile.version != version:
        raise ValidationError("agent_profile_unavailable", "The saved assistant profile version is unavailable; start a new run.")
    return profile


def permits(profile: AgentProfile, tool) -> bool:
    return tool is not None and (tool.canonical_name or tool.name) in profile.allowed_tools
