# YoloWebAgentStarter 源代码快照

> ## 公开发布状态：阻断
>
> 本文件记录的固定上游树没有可核验的顶层 `LICENSE` 或 `NOTICE`。因此，当前仓库中的 MIT `LICENSE`/`NOTICE` **不能**单独证明从该快照选择性派生的代码可被再许可或公开发布。公开 tag 必须先满足 [上游授权发布门槛](docs/provenance/UPSTREAM_AUTHORIZATION.md)；详细的逐模块来源审计见 [migration_matrix.md](migration_matrix.md)。在该证据获得并由维护者填写为 `approved` 前，禁止发布或宣称这是一个已获授权的开源衍生版本。

## 1. 固定迁移基线

| 字段 | 值 |
|------|----|
| Source repository | YoloWebAgent source repository (not included here) |
| Remote | `https://github.com/fengzhizi715/YoloWebAgent.git` |
| Branch at capture | `feature/mysql` |
| Commit SHA | `701f6e5a63b73f39e35f48fb6de7d2414401875a` |
| Tree SHA | `84860c53f38c160ce0a13a100f48be2074860198` |
| Author | `fengzhizi715` |
| Commit time | `2026-08-05T11:38:17+08:00` |
| Subject | `Update ImportCenterModal.tsx` |
| Working tree | clean |
| Captured at | `2026-08-08` |

该完整 commit SHA 是 Community v2 唯一迁移基线。后续源项目提交不会自动进入 Starter；任何基线变更都必须更新本文件和 `migration_matrix.md`，并重新获得可核验的许可证据。

## 2. 固定方式

- 仅在 Starter 文档中记录完整 SHA 和 tree SHA，本阶段不在源仓库创建 tag，不修改源仓库状态。
- Phase 2 初始化 Starter Git 时，从干净空仓库开始，不复制源项目 `.git`。
- 迁移时从该 commit 读取文件，例如：

```bash
git -C /path/to/yolowebagent-source show 701f6e5a63b73f39e35f48fb6de7d2414401875a:path/to/file
```

- 不从源项目工作树盲目复制未提交内容。
- 如必须采用新源 commit，应先比较两个 commit 的模块差异和公开风险，再显式批准新的基线。

## 3. 源项目规模快照

| 范围 | 规模 |
|------|------|
| Backend Python | 约 404 个文件、39.5k 行 |
| Frontend TS / TSX | 约 294 个文件、49.2k 行 |
| Alembic migrations | 19 个 |
| Backend tests | 64 个 |
| Dataset domain | 约 64 个 Python 文件、8.4k 行 |
| Training domain | 约 53 个 Python 文件、4.3k 行 |
| Deployment domain | 约 53 个 Python 文件、4.1k 行 |
| Agent domain | 约 37 个 Python 文件、8.5k 行 |

这些规模用于估算，不表示对应目录可以整目录迁移。

## 4. 基线风险

- 基线来自 `feature/mysql`，包含 MySQL、商业授权、Agent、Workflow、复杂 Deployment 和数据准备等 Starter 排除能力。
- `main.py` 和 API router 聚合入口会加载大量商业模块。
- `core/models.py` 混合 Community 与 Enterprise 表。
- training runtime、YOLO engine、model service、annotation UI 都存在跨域依赖。
- 前端路由、API client、类型和多项核心页面混有 Auth、License 和商业功能。

因此本基线只能作为选择性抽取来源，禁止通过复制完整目录后简单隐藏菜单来形成公开版本。它也不是授权证据；在发布门槛满足前，不得把其内容作为可再许可的开源来源使用。

## 5. Community v2 选择性抽取记录

在用户确认扩展范围后，Community v2 从同一固定 commit 仅参考并重建了以下公开工作流契约：

- `backend/app/core/task_types.py`、`backend/app/core/schemas.py` 与 `backend/app/dataset/annotation/service.py` 的 OBB / classify task 与标注模型；
- `backend/app/dataset/exchange/yolo/dataset_file_writer.py`、`label_writer.py` 的 OBB 八点 YOLO 标签与 classify 目录布局；
- `backend/app/sam/schemas.py`、`backend/app/sam/backends/box_stub.py`、`backend/app/sam/ultralytics_backend.py` 的框/点提示、建议多边形和延迟加载模型边界。
- `backend/app/evaluation/service.py`、`runner.py`、`artifacts.py`、`error_samples.py` 与前端评估详情面板的本地 `yolo val`、受管产物、日志和错误样本契约；Starter 仅保留 detect、segment、OBB、classify 范围。
- `backend/app/training/runtime/device_service.py` 的本地设备发现、CUDA 选择和多 GPU `device=0,1` 契约；Starter 将其重建为本地单机服务，不引入上游资源调度器。
- `backend/app/settings/service.py`、`schemas.py` 与 `logs/service.py` 的可见工作流契约；Starter 只提供独立的本地 SAM 设置、语言偏好和后端运行日志边界。

Starter 对这些能力保留独立的 SQLite schema、存储边界、路由和前端实现；不复制 Enterprise 的设置/Profile 引擎、Vision Provider、Agent、自动标注、文本分割或 deployment 模块，也不在运行时读取源仓库。SAM 设置和运行日志属于 Starter 的最小本地边界，不依赖 Enterprise 设置/Profile 运行时。

## 6. 快照验证命令

```bash
git -C /path/to/yolowebagent-source status --porcelain=v1
git -C /path/to/yolowebagent-source rev-parse HEAD
git -C /path/to/yolowebagent-source rev-parse HEAD^{tree}
git -C /path/to/yolowebagent-source show -s --format='%H%n%T%n%aI%n%s' HEAD
```

期望值：

```text
HEAD  = 701f6e5a63b73f39e35f48fb6de7d2414401875a
TREE  = 84860c53f38c160ce0a13a100f48be2074860198
status output is empty
```
