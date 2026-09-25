"""Read-only composite reports over existing domain-backed handlers."""

from app.agent.bounds import bound_json
from app.core.errors import DomainError


def diagnostic_handlers(handlers):
    def read(name, session, arguments):
        try:
            return handlers[name][2](session, arguments)
        except DomainError as exc:
            return {"error": exc.error_code}

    def readiness(session, arguments):
        facts = {name: read(name, session, arguments) for name in (
            "dataset_summary", "dataset_quality_report", "dataset_validate",
        )}
        dataset = facts["dataset_summary"]
        validation = facts["dataset_validate"]
        quality = facts["dataset_quality_report"]
        issues = []
        missing = [name for name, result in facts.items() if result.get("error") or result.get("truncated") is True]
        stats = validation.get("summary") or validation
        errors = stats.get("error_count")
        if dataset.get("image_count") == 0:
            issues.append("数据集没有图片。")
        if dataset.get("class_count") == 0:
            issues.append("没有配置类别。")
        if dataset.get("annotated_image_count") == 0:
            issues.append("没有已标注图片，不能据此开始监督式训练。")
        splits = dataset.get("split_counts") or {}
        incomplete = any(dataset.get(key) is None for key in (
            "image_count", "class_count", "annotated_image_count",
        ))
        for split in ("train", "val"):
            if splits.get(split) is None:
                incomplete = True
            elif splits[split] == 0:
                issues.append(f"持久化 {split} split 为空，需先人工检查划分。")
        if isinstance(errors, int) and errors > 0:
            issues.append(f"校验发现 {errors} 个错误。")
        blocked = bool(issues)
        coverage = (quality.get("summary") or {}).get("coverage")
        incomplete = incomplete or coverage is None or errors is None or bool(missing)
        if isinstance(coverage, (int, float)) and coverage < 1:
            issues.append("存在未标注图片；请核对负样本与标注缺口，不要把未标注都当作负样本。")
        if incomplete:
            issues.append("部分证据不足，无法确认训练准备度。")
        status = "blocked" if blocked else "unknown" if incomplete else "review_required"
        return bound_json({
            "dataset_id": arguments.get("dataset_id"), "status": status,
            "summary": {"blocked": "存在训练前置问题。", "unknown": "证据不足，需补充检查。",
                        "review_required": "已读取基础证据，训练前仍需人工核对类别、标注与 split。"}[status],
            "evidence": [{"tool": name, "facts": bound_json(result, limit=2500)} for name, result in facts.items()],
            "issues": issues,
            "next_steps": ["先处理校验错误和空 split，再复查标注覆盖与类别分布。", "本报告不创建任务；确认训练参数后再人工提交。"],
        })

    def failure(session, arguments):
        task = read("training_status", session, arguments)
        logs = read("training_logs", session, arguments)
        text = f"{task.get('error_message') or ''}\n{logs.get('preview') or ''}".lower()
        hypotheses = []
        rules = [
            (("out of memory", "cuda oom", "mps backend out of memory"), "memory_pressure", "日志含内存/显存不足信号。", "核对设备可用内存，逐步降低 batch 或输入尺寸后人工重试。"),
            (("no module named", "modulenotfounderror"), "missing_dependency", "日志含依赖缺失信号。", "检查训练进程使用的虚拟环境及项目依赖。"),
            (("filenotfounderror", "no such file", "dataset not found"), "data_or_artifact_path", "日志可能涉及数据或产物路径。", "检查受管数据导出与产物是否完整，不要使用外部权重路径。"),
            (("no labels found", "no images found", "dataset is empty"), "empty_training_data", "日志含训练数据或标签缺失信号。", "查询数据集校验与持久化 split，核对有效标注。"),
        ]
        for tokens, code, explanation, action in rules:
            matched = [token for token in tokens if token in text]
            if matched and task.get("status") == "failed":
                hypotheses.append({"code": code, "confidence": "possible", "evidence": matched,
                                   "explanation": explanation, "next_step": action})
        issues = ["规则匹配仅表示可能原因，不排除其他问题。"]
        if not logs.get("preview") or logs.get("error"):
            issues.append("日志不可用或为空；仅凭任务报错不能确认根因。")
        else:
            issues.append("只读取受限日志尾部，并非完整日志。")
        if task.get("status") != "failed":
            issues.append("没有证据表明当前任务处于失败状态，不进行失败根因断言。")
        return bound_json({
            "task_id": task.get("id") or arguments.get("task_id") or arguments.get("training_task_id"),
            "dataset_id": task.get("dataset_id"), "status": task.get("status", "unknown"),
            "summary": f"识别到 {len(hypotheses)} 项可能原因。" if hypotheses else "当前证据不足以归因。",
            "hypotheses": hypotheses,
            "evidence": [{"tool": "training_status", "facts": bound_json(task, limit=2500)}, {"tool": "training_logs", "facts": bound_json(logs, limit=2500)}],
            "issues": issues,
            "next_steps": [item["next_step"] for item in hypotheses] or ["补充状态、日志及数据校验结果后再确定处理方案。"],
        })

    return {
        "dataset_readiness": ("Read-only training-readiness report with validation, quality, persisted splits and evidence.", handlers["dataset_summary"][1], readiness),
        "training_diagnose_failure": ("Read-only training failure diagnosis: evidence, possible causes and manual next steps, never retries tasks.", handlers["training_status"][1], failure),
    }
