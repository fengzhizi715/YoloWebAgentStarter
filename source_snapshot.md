# YoloWebAgentStarter 源代码快照

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

该完整 commit SHA 是 Starter v1 唯一迁移基线。后续源项目提交不会自动进入 Starter；任何基线变更都必须更新本文件和 `migration_matrix.md`。

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

因此本基线只能作为选择性抽取来源，禁止通过复制完整目录后简单隐藏菜单来形成公开版本。

## 5. 快照验证命令

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
