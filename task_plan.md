# YoloWebAgentStarter 迁移计划

## Goal

从固定的 YoloWebAgent 源快照选择性抽取核心能力，建立一个可独立安装、运行和公开发布的开源引流产品；Starter 不运行时依赖 Enterprise，Enterprise 也不依赖 Starter 的仓库、分支或发布节奏。

## Product Positioning

YoloWebAgentStarter 是本地 YOLO 数据集与训练工作台，必须独立跑通：

```text
图片 → 数据集 → 人工标注 → 校验 → 本地 YOLO 训练 → best.pt / last.pt → ONNX
```

Starter 提供完整的基础闭环；Enterprise 提供自动化、协作、诊断和部署交付能力。

## Migration Principles

1. **选择性抽取**：从固定的源项目 commit 按功能切片迁移，不复制完整项目后再大规模删除。
2. **干净公开历史**：Starter 建立新的 Git 历史，不复制源项目 `.git`，不带入商业代码和内部历史。
3. **垂直切片交付**：每个功能阶段同时完成后端、前端、数据库和测试，不采用“全部后端完成后再做全部前端”的顺序。
4. **独立运行**：Starter 不读取 Enterprise 的数据库、环境、文件目录、Python 包或前端构建产物。
5. **弱耦合**：两个项目只在 YOLO 格式、TaskType、模型元数据和公开文档层保持一致。
6. **最小维护承诺**：优先保证安装、基础闭环和严重问题修复，不为引流版本建立复杂扩展框架。

## Starter v1 Scope Matrix

| 能力 | v1 状态 | 说明 |
|------|---------|------|
| detect 标注 / 导出 / 训练 | included | Starter 核心闭环 |
| segment 标注 / 导出 / 训练 | included | Starter 核心闭环 |
| OBB | excluded | 推迟到后续版本，不进入 Starter v1 |
| classify | excluded | 推迟到后续版本，不进入 Starter v1 |
| pose | excluded | 不进入 Starter v1 |
| 图片目录导入 | included | 本地文件系统 |
| YOLO 导入 / 导出 | included | 作为主要数据交换方式 |
| native archive | excluded | v1 只使用 YOLO 作为公开数据交换格式 |
| split 与基础校验 | included | 训练前置能力 |
| 视频抽帧 / tiling / 去重 / 高级质量分析 | excluded | 属于高级数据准备 |
| 本地训练任务、日志、状态和 checkpoint | included | 不接 Workflow / Evaluation / Iteration |
| 模型版本、PT 下载、ONNX 导出 | included | ONNX 仅保留 Starter 所需最小能力 |
| TensorRT / OpenVINO / INT8 / Benchmark | excluded | Enterprise 部署能力 |
| Evaluation / Active Learning / Agent / Workflow | excluded | Enterprise 闭环能力 |
| SAM / 自动标注 / VLM | excluded | Enterprise 自动化能力 |
| RBAC / 商业 License / Tenant | excluded | Starter 默认为本地单用户产品 |

## Current Phase

Phase 5 已完成，下一阶段进入 Phase 6：公开发布、安全审计与引流体验。

## Phases

### Phase 1: 产品边界、迁移基线与文件清单

- [x] 确认 Starter v1 只支持 detect / segment，OBB / classify / pose 不进入 v1。
- [x] 确认仅保留 YOLO import / export 与基础 validation，排除 native archive、质量报告和模型快速测试。
- [x] 确认 Starter 为无登录本地单用户模式，默认只监听 `127.0.0.1`。
- [x] 确认首发正式支持 macOS / Linux，Windows 暂不承诺。
- [x] 确认维护策略为基础修复维护，不承诺长期高频迭代。
- [x] 固定源项目 commit SHA，作为唯一迁移基线。
- [x] 创建迁移矩阵，逐项列出 module / API / table / page / test 的 keep / rewrite / exclude。
- [x] 固化 Starter API、数据目录、YOLO 格式、模型元数据和产物命名约定。
- [x] 建立第三方依赖、模型权重、素材、文档和许可证审计清单。
- **Exit gate:** 功能矩阵、迁移矩阵、源 commit 和发布约束全部确认。
- **Status:** complete

### Phase 2: 干净仓库与基础设施

- [x] 初始化新的 Git 历史，不复制源项目 `.git`。
- [x] 创建 README、LICENSE、NOTICE、CHANGELOG、贡献说明和维护策略。
- [x] 建立 backend / frontend 目录、运行脚本和安全的环境变量模板。
- [x] 迁移最小 core：数据库、存储、路径、ID、时间、错误、TaskType、日志和配置。
- [x] 建立 Starter 专用 API 聚合入口和最小前端壳。
- [x] 创建 Starter 全新的 baseline migration，不复制 Enterprise 的 19 个历史 migration。
- [x] 只建立 Community 核心表，不包含 Deployment、Evaluation、Agent、Workflow、License 等表。
- [x] 默认仅绑定 `127.0.0.1`，并记录无认证模式的安全边界。
- [x] 建立最小依赖清单和基础 CI 骨架。
- [x] 记录源 commit、抽取日期和已知限制。
- **Exit gate:** 全新环境可启动空白后端和前端，数据库可从零创建。
- **Status:** complete

### Phase 3: 数据集与标注垂直切片

- [x] 迁移 Dataset、ImageItem、ClassLabel、Annotation 和 split 数据模型。
- [x] 迁移图片目录导入、数据集 CRUD、类别管理和图片读取。
- [x] 迁移 Phase 1 确认的人工标注类型及坐标转换。
- [x] 迁移基础数据集校验和 YOLO 导入 / 导出。
- [x] 迁移数据集列表、详情、图片列表和标注页面。
- [x] 重建对应 API client、前端类型、错误处理和空状态。
- [x] 移除 SAM、AI 自动标注、VLM、Active Learning 和高级 Preparation 入口。
- [x] 补齐数据集、标注、校验、YOLO 往返和前端构建测试。
- **Exit gate:** 用户可完成“导入图片 → 建类 → 标注 → 校验 → YOLO 导出”。
- **Status:** complete

### Phase 4: 本地 YOLO 训练垂直切片

- [x] 迁移训练配置、任务队列、runner、进程控制、日志、checkpoint 和训练摘要。
- [x] 清理 training runtime 对 Workflow、Evaluation 和 Iteration 的直接依赖。
- [x] 保留 TaskType 与权重族匹配校验。
- [x] 迁移训练中心、训练任务列表、训练任务详情和日志查看。
- [x] 移除 Agent 入口、自动评估触发和闭环跟踪入口。
- [x] 验证任务创建、排队、运行、停止、失败恢复和产物路径。
- [x] 为 Phase 1 确认的每个 TaskType 补齐训练测试。
- **Exit gate:** 用户可从 Starter 数据集发起本地训练并获得 best.pt / last.pt。
- **Status:** complete

### Phase 5: 模型管理与 PT / ONNX 垂直切片

- [x] 迁移 ModelVersion 核心模型和训练产物入库逻辑。
- [x] 迁移模型列表、详情、备注、归档、恢复、删除和下载。
- [x] 明确 PT “导出”为训练产物下载或已导入 PT 下载，不创建伪转换任务。
- [x] 从完整 Deployment 中抽取最小 ONNX 导出能力。
- [x] 清理 YOLO engine 对 FP16、量化、OpenVINO 和 TensorRT 模块的直接依赖。
- [x] 清理模型服务对自动标注解析器、部署指标和商业模块的直接依赖。
- [x] 迁移模型列表、详情、PT 下载和 ONNX 导出入口。
- [x] 补齐 PT 下载、ONNX 生成、ONNX 入库、重复导出和失败处理测试。
- **Exit gate:** 用户可管理训练模型、下载 PT，并从 PT 生成和下载 ONNX。
- **Status:** complete

### Phase 6: 公开发布、安全审计与引流体验

- [x] 在独立虚拟环境和全新数据目录完成安装验证。
- [x] 审计 Python / npm 依赖、模型权重、vendor wheels、素材和截图的公开许可与分发方式。
- [x] 扫描并移除商业 License、内部部署脚本、客户信息、密钥、内网地址、日志和本地绝对路径（来源快照仅保留脱敏归档信息）。
- [x] 验证文件上传、图片读取、模型下载和导出路径不能越过 Starter 管理目录。
- [x] 明确无认证模式只适用于本地使用，不宣称可直接公网部署。
- [x] 编写 3～5 分钟 Quick Start、配置说明、数据目录、故障排查和已知限制。
- [x] 提供可生成的小型 YOLO 示例数据集。
- [ ] 在 README 中加入 Starter / Enterprise 功能对比和已批准的 Enterprise 联系入口。
- [x] 建立 backend test、frontend test/build 的 CI；完成真实 CPU YOLO/ONNX smoke，GPU/MPS 保持发布前的可选平台门禁。
- **Exit gate:** 许可证、安全、秘密信息、安装体验和引流入口全部通过发布检查。
- **Status:** in progress — waiting for an approved Enterprise contact URL before a marketing release.

### Phase 7: Release Candidate 与迁移收尾

- [ ] 验证完整链路：创建数据集 → 导入图片 → 标注 → 校验 → 训练 → PT → ONNX。
- [ ] 验证 YOLO 导入 / 导出往返和 Starter 数据目录独立性。
- [ ] 验证 Starter 不 import Enterprise，不读取 Enterprise 数据库、环境和产物目录。
- [ ] 在 Phase 1 确认的每个支持平台执行安装和 smoke test。
- [ ] 发布候选版本，记录未纳入能力和维护范围。
- [ ] 固化版本号、发布说明、问题反馈和 Enterprise 转化入口。
- [ ] Enterprise 保持独立产品源头；社区修复仅在人工评估后选择性回收。
- **Exit gate:** Release Candidate 验收通过，可创建公开首发版本。
- **Status:** pending

## Database Strategy

- Starter 使用自己的全新 baseline migration。
- 核心表只覆盖 Dataset、ImageItem、ClassLabel、Annotation、TrainingTask、TrainingProfile、ModelVersion 及必要任务日志/状态。
- 不要求兼容或直接升级 Enterprise SQLite / MySQL 数据库。
- 用户数据迁移通过 Phase 1 最终确认的公开交换格式完成，默认以 YOLO 导入 / 导出为主。
- Starter 文件路径不得引用源项目数据目录；测试必须使用独立临时目录。

## Release Blockers

以下任一项未完成，都不能公开发布：

- Phase 1 决策门槛未关闭。
- Starter 仍直接 import Enterprise 专属模块。
- 新环境无法从零创建数据库或完成完整闭环。
- 存在商业 License、客户信息、密钥、内部地址或私有构建资源。
- 第三方依赖、模型权重、素材或截图的公开分发条件未核对。
- 无认证服务默认监听公网地址，或文件接口存在目录越界风险。
- README 缺少安装步骤、限制说明和维护范围。

## Resolved Phase 1 Questions

| 问题 | 已锁定答案 |
|------|------------|
| TaskType | 只支持 detect / segment；OBB / classify / pose 排除 |
| 数据交换与质量能力 | 保留 YOLO import / export 和基础 validation；native archive、COCO、质量报告排除 |
| 运行模式 | 本地无登录、单用户，默认绑定 `127.0.0.1` |
| 平台与设备 | macOS / Linux 正式支持；CPU 必测，CUDA / MPS best-effort；Windows 暂不承诺 |
| 维护承诺 | 基础修复维护，主要处理安装阻断、严重缺陷和安全问题 |

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Starter 与 Enterprise 不做运行时或仓库依赖 | Starter 主要承担引流，未来不一定长期维护。 |
| Enterprise 保持主要产品源头 | 商业路线和授权逻辑不受 Starter 发布节奏约束。 |
| 采用固定 commit 的选择性抽取 | 降低商业代码泄漏、删除遗漏和公开历史污染风险。 |
| 建立新的 Git 历史 | Starter 是独立公开产品，不复制私有历史。 |
| 按数据集、训练、模型三个垂直切片迁移 | 每个阶段都能端到端验证 API、UI、数据和测试。 |
| Starter 使用全新 baseline migration | 不继承 Enterprise 表和历史迁移负担。 |
| 只抽取最小 ONNX 能力 | TensorRT、OpenVINO、量化和部署包属于 Enterprise。 |
| 无认证时默认仅监听 localhost | 限定 Starter 的安全使用边界。 |
| 发布审计是阻断条件 | 开源仓库不得包含商业代码、敏感信息或未经核对的分发资源。 |
| 源迁移基线固定为 `701f6e5a63b73f39e35f48fb6de7d2414401875a` | 后续源提交不会隐式进入 Starter，确保迁移可复现。 |
| Starter v1 只支持 detect / segment | 控制标注、导出、训练和测试矩阵，避免首发范围膨胀。 |
| v1 仅保留 YOLO 数据交换和 ONNX FP32 | 满足基础闭环，排除 native archive、COCO 和高级部署能力。 |
| 首发支持 macOS / Linux，采用基础修复维护 | 与开源引流定位和可投入维护资源匹配。 |

## Estimated Effort

- 内部可运行快照：约 12～18 人日。
- 可公开发布的 Starter MVP：约 22～32 人日。
- 多平台、文档和安装体验完善版本：约 30～45 人日。
- 如果 Phase 1 追加 OBB 和 classify：额外约 5～8 人日。

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| 目标目录目前为空且还不是 Git 仓库 | 1 | 将干净 Git 初始化纳入 Phase 2；不复制源项目历史。 |

## Notes

- 源项目：固定快照，见 `source_snapshot.md`
- Starter 项目：本仓库
- 当前计划只定义迁移范围，不开始复制或修改业务代码。
- 每完成一个阶段，同步更新本文件和发布检查记录。
- Phase 1 已锁定；任何范围变更必须显式更新 `phase1_scope.md`、`migration_matrix.md`、工作量和测试门槛。
- Phase 2 才初始化 Starter Git 和项目骨架；Phase 1 不复制或修改业务代码。
