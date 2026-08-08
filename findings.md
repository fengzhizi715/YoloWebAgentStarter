# YoloWebAgentStarter 迁移发现

## Requirements

- 创建一个独立的 YoloWebAgentStarter 开源项目。
- Starter 用于引流，未来不保证长期完整维护。
- Starter 不应成为 Enterprise 的运行时或仓库依赖。
- Starter 的核心功能为：标注、数据集管理、YOLO 训练、模型管理、PT / ONNX 导出。
- 当前 Phase 1 只创建并锁定功能矩阵、迁移清单和源快照，不修改业务代码。

## Research Findings

### Phase 2 / Phase 3 实施发现

- 固定源快照中的四张核心表仍带有 Enterprise 字段：Dataset 含 pose/preparation/tenant 字段，ImageItem 含视频与 lineage 字段，Annotation 含 OBB/pose/confidence 字段；Starter baseline 应重新定义最小表，而不是复制 ORM 后删列。
- Starter Phase 3 的最小持久化模型确定为 `datasets`、`class_labels`、`image_items`、`annotations`；split 直接持久化在 `image_items.split`，数据集 task type 只允许 `detect | segment`。
- 浏览器无法安全、可移植地提交任意服务端目录。Starter 同时提供 multipart 图片上传与受 `YWA_IMPORT_ROOT` 限制的目录扫描；扫描路径只能解析到该根目录内。
- Starter 图片统一复制到自己的受管数据目录，数据库不保存对 Enterprise 或任意外部目录的运行时依赖。
- 标注 API 使用整图 replace 语义，并在事务内校验 class、annotation family、绝对坐标和图片尺寸；前端 Canvas 临时状态不写入数据库。
- YOLO import 首版采用 ZIP 上传，YOLO export 返回 ZIP；两者只接受 detect / segment，并复用 `image_items.split`。
- 前端采用独立的轻量 React/Vite 壳和 Konva 标注画布，不迁移 Enterprise 的 AuthProvider、License gating、Agent、SAM、OBB、Pose 或大体量全局样式。
- 训练和模型属于 Phase 4/5；本轮只在产品导航中展示后续阶段状态，不创建伪 API 或不可用操作。

### Phase 4 实施发现

- Starter 训练不能直接迁移源项目的 `TrainingTask`，因为源模型混入 Workflow、Evaluation、Iteration、Auth/RBAC、ModelVersion 和部署字段；Starter 使用独立的最小 training profile/task 表。
- 训练数据必须从 Starter 受管图片和绝对像素标注重新物化为任务目录下的 YOLO 数据集，并沿用 `image_items.split`，不能在训练启动时偷偷随机覆盖 split。
- Ultralytics 运行时只通过可替换的命令前缀启动；默认定位仓库 `.venv/bin/yolo`，也支持 `YWA_YOLO_EXECUTABLE` 用于确定性 smoke test 和受控环境。
- 本地单用户版本采用单进程 FIFO 队列；进程组停止、数据库 `stop_requested` 标记和启动时 running→failed 恢复共同覆盖服务重启与停止竞态。
- Phase 4 完成条件要求 `best.pt` 和 `last.pt` 同时存在；进程返回 0 但 checkpoint 缺失时任务仍标记 failed，避免产生不可用的“成功”任务。

### Phase 5 实施发现

- ModelVersion 只记录 Starter 训练任务产生的 best/last PT 和其 ONNX 导出；不提供任意外部 PT 导入入口，避免数据库指向不可控文件。
- 训练完成后将 best.pt / last.pt 复制到 `data/models/<model_id>/` 并写入模型表；训练运行目录仍保留为可追溯的任务产物。
- ONNX 导出只接受受管 Ultralytics PT，固定 FP32、640、batch 1、静态 shape，不迁移 FP16、INT8、TensorRT 或 OpenVINO 转换链。
- ONNX 记录通过 `source_model_id` 关联源 PT；重复导出直接返回已有记录，转换失败时回滚记录并保留源 PT。

### Phase 1 固定源快照

- 源仓库：`/Users/tony/PycharmProjects/YoloWebAgent`
- 分支：`feature/mysql`
- Commit：`701f6e5a63b73f39e35f48fb6de7d2414401875a`
- Tree：`84860c53f38c160ce0a13a100f48be2074860198`
- 提交时间：`2026-08-05T11:38:17+08:00`
- 提交说明：`Update ImportCenterModal.tsx`
- 检查时工作树为空，无未提交修改；该 commit 可作为可复现迁移基线。

### Phase 1 API 与数据表发现

- Starter API 应从现有聚合路由中只保留 system、datasets、training、models 的必要端点，重新建立专用聚合入口。
- Auth、license、agent、workflow、sam、vision、vision-language、AI annotation、deployment、evaluation、active learning、auto annotation、import center、preparation 和 runtime logs 默认排除。
- `datasets.py` 中 native archive、COCO bbox export、quality report 属于可选或排除能力，不能整文件原样迁移。
- `models.py` 中 compare、quick test、save-test 不属于默认 Starter 模型管理范围，需从基础 CRUD / download / export-onnx 中分离。
- 数据库核心候选表为 datasets、class_labels、image_items、annotations、training_tasks、training_profiles、model_versions。
- users、permissions、role_permissions、audit_logs 以及所有 AI、部署、评估、主动学习、Agent、Workflow 表默认排除。
- Starter v1 采用同步图片目录扫描，不保留 ImportJob；若未来引入大规模异步导入，应重新设计独立任务模型，而不是复制 Import Center。

### Phase 1 后端依赖发现

- `main.py` 直接依赖 Auth、Deployment capability 和 SAM warmup，Starter 必须重写启动入口。
- `training/runtime/runner.py` 直接依赖 Workflow、Evaluation、Iteration，训练切片不能直接复制 runner。
- `yolo/engines/ultralytics.py` 直接依赖 ONNX FP16、量化、OpenVINO、TensorRT 和 Deployment schema，需收缩为训练 + 基础 ONNX 导出接口。
- `models/service.py` 和 `training/application/dataset_training_service.py` 读取当前请求用户，单用户 Starter 需要改成无用户字段或固定本地 actor，而不是迁移 Auth。
- 数据集 service、YOLO importer/writer、annotation service 和 validator 存在 pose schema 依赖；既然 Starter v1 排除 pose，这些依赖需要移除或改成仅 detect/segment 分支。
- `dataset/annotation/ai`、`dataset/annotation/auto`、`dataset/preparation`、`dataset/quality` 默认不迁移；`dataset/validation` 与 `dataset/exchange/yolo` 选择性迁移。

### Phase 1 前端依赖发现

- `main.tsx` 集中处理 Agent、Workflow、Evaluation、Deployment、Active Learning、Auto Annotation、License 等路由，Starter 应重建精简入口而不是原样迁移。
- `api/client.ts` 混合所有产品 API，Starter 应拆成 dataset、training、models/system 等小型 client，避免删除遗漏。
- 标注目录同时包含 OBB、Pose、SAM 和 AI Suggestions；Starter 只迁移 AnnotationCanvas、BBoxRect、PolygonShape、基础列表/工具栏及必要 utilities。
- 保留页面候选为 HomePage、AnnotationPage、TrainingControlCenterPage、TrainingTaskListPage、TrainingTaskDetailPage、TrainingProfilePage、ModelListPage、ModelDetailPage。
- Login、Profile、Users、Settings、Agent、Workflow、Evaluation、Deployment、Active Learning、Auto Annotation、Preparation、Quality、Logs 页面默认排除。
- ModelDetail 和 ModelList 内仍混入评估、部署包、对比和 License gating，需重写为基础元数据、训练来源、PT 下载和 ONNX 导出视图。
- 共享 UI primitives、dataset-feature、training-* 组件可选择性迁移；Agent、workflow、deployment、evaluation、preparation、users 组件目录整体排除。

### Phase 1 依赖与测试发现

- Starter 后端基础依赖候选：FastAPI、Uvicorn、Pydantic、SQLAlchemy、Alembic、python-multipart、Pillow、PyYAML、NumPy、OpenCV、Ultralytics、ONNX 相关依赖和 HTTPX。
- PyMySQL、PyJWT、bcrypt、paramiko、SAM 可选依赖和部署运行时默认排除；是否保留 HTTPX 需由迁移后的实际 import 决定。
- ONNX FP32 导出当前使用 onnx、onnxruntime、onnxscript、onnx_ir、ml_dtypes；迁移时应通过最小导出 smoke test确认真实依赖，避免直接照搬完整 requirements。
- 前端可保留 React、React DOM、Konva、react-konva、lucide-react、Vite、TypeScript、Vitest；React Flow 和 AJV 主要服务 Workflow，默认排除。
- 后端测试保留/改写候选：app config、core paths、errors、time、image scanning、dataset validation、segmentation polygon、YOLO IO、training、training center、training command builder、dataset training jobs、model versions、ultralytics engine、YOLO CLI。
- OBB、classify、pose、native exchange、quality、import center 测试是否迁移取决于功能矩阵；默认不进入 v1 测试集。
- Auth、License、MySQL、Agent、Workflow、Evaluation、Deployment、Active Learning、Auto Annotation、SAM、VLM、Preparation 和商业 release 测试整体排除。
- 前端测试应选择性保留 annotation utilities、API 基础请求、UI primitives、菜单导航、时间和模型/训练纯 helper；商业模块测试整体排除。

### Phase 1 文件级迁移发现

- training 目录同时保留根级兼容入口和 application/config/runtime/artifacts/observability 新分层实现；Starter 应优先迁移分层实现，只在确有调用方时保留薄兼容入口，避免复制两套逻辑。
- core/models.py、core/schemas.py、datasets.py、models.py、training.py 都需要按 Starter 边界重写，不能以 keep 方式原样复制。
- dataset/annotation 的 converters/families/service 可作为基础，但需要删除 pose、AI、auto annotation 分支。
- dataset/exchange/yolo 的 exporter/importer/writer/planner 是核心候选，但需要去除 pose/classify/OBB 分支或只保留 Phase 1 最终支持的 TaskType。
- frontend AnnotationPage、AnnotationCanvas、AnnotationToolbar、AnnotationListPanel、annotationApi 和 annotationTypes 都混有 Auth、License、SAM、AI、OBB、Pose、classify 逻辑，迁移状态应为 rewrite。
- HomePage 混有 Import Center、License 和权限入口；TrainingControlCenterPage 混有 Workflow 和权限入口；ModelListPage/ModelDetailPage 混有部署、评估、Agent、比较和测试入口，均需重写。
- 前端共享 UI primitives 和时间/导航纯 helper 可以 keep-with-review；菜单、Topbar、Sidebar、API client 和总类型文件应重写为 Starter 专用版本。

### 目标仓库状态

- `/Users/tony/PycharmProjects/YoloWebAgentStarter` 已包含 Phase 1 的计划、功能矩阵、迁移清单和源快照文档。
- 目标目录尚未初始化 Git，也没有项目级 `AGENTS.md`；这两项保留到 Phase 2。
- 迁移计划应放在目标项目根目录，作为后续执行时的持久化工作记忆。

### 源仓库规模

- 源项目当前分支为 `feature/mysql`，最近提交为 `701f6e5 Update ImportCenterModal.tsx`。
- Backend 约 404 个 Python 文件、3.95 万行。
- Frontend 约 294 个 TypeScript / TSX 文件、4.92 万行。
- Alembic migration 约 19 个，后端测试约 64 个。

### 主要领域规模

- `backend/app/dataset`：约 64 个 Python 文件、8.4k 行，是 Starter 的最大核心领域。
- `backend/app/training`：约 53 个文件、4.3k 行。
- `backend/app/deployment`：约 53 个文件、4.1k 行，但大部分超出 Starter 范围。
- `backend/app/agent`：约 37 个文件、8.5k 行，应排除在 Starter 之外。
- 前端现有页面约 35 个，包含 Agent、Workflow、Evaluation、Deployment、Active Learning、SAM 等商业/扩展页面。

### 已确认的耦合点

- `backend/app/api/routes.py` 会统一加载所有领域路由。
- `backend/app/main.py` 启动时会初始化 License、Deployment capability、SAM warmup，以及导入/准备任务恢复。
- `backend/app/api/routers/datasets.py` 和 `backend/app/api/routers/models.py` 混合导入 Agent、Deployment、Evaluation、Iteration、SAM、自动标注等模块。
- `backend/app/training/runtime/runner.py` 直接依赖 Workflow、Evaluation、Iteration。
- `backend/app/yolo/engines/ultralytics.py` 直接依赖 ONNX FP16、ONNX quantization、OpenVINO 和 TensorRT 转换模块。
- `backend/app/core/models.py` 混合定义数据集、训练、模型、部署、评估、Active Learning、Agent、Workflow 等表。
- `backend/app/models/service.py` 仍依赖自动标注结果解析器和导出指标逻辑，需要清理成 Starter 可独立使用的模型服务。
- 前端 `main.tsx`、`menuConfig.ts`、`routeAccess.ts` 和模型部署组件中存在 License gating 和商业功能入口。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 采用独立 Starter 快照，而不是 Enterprise 依赖 Starter | 符合引流产品定位，也避免 Starter 未来停止维护后影响 Enterprise。 |
| Starter 初版聚焦图片、标注、训练和模型产物 | 形成可验证的完整闭环，减少商业扩展和硬件部署复杂度。 |
| ONNX 作为 Starter 的唯一模型转换能力 | 用户需要基本交付能力，但 TensorRT/OpenVINO/量化属于商业部署能力。 |
| 在每个垂直切片内拆 API 和领域依赖 | 当前路由和 startup import 是最明显的跨域耦合点，但应和对应前端及测试一起验证。 |
| 历史数据只通过 YOLO 交换，v1 排除 native archive | 不把源项目 SQLite、本地文件路径或 Enterprise 项目包作为隐式依赖。 |

### 计划评审后的补充发现

- 原计划按“全部后端 → 全部前端”推进，接口和页面问题会过晚暴露；应改为数据集/标注、训练、模型/导出三个垂直切片。
- 迁移方式必须明确为从固定 commit 选择性抽取，不能复制完整仓库后依赖人工删除商业能力。
- Starter 应建立新的 Git 历史，避免公开源项目的内部提交历史和已删除内容。
- Starter 不需要复用 Enterprise 的 19 个历史 migration；新项目更适合一份干净的 baseline migration。
- 无登录 Starter 必须默认绑定 localhost，并明确不直接支持公网部署。
- 许可证、模型权重、vendor wheels、素材、截图和秘密信息审计应是发布阻断条件，而不是普通备注。
- “用于引流”要求 README、示例数据、截图/演示、版本对比和 Enterprise 联系入口进入验收范围。
- 原 15～25 人日适合作为内部可运行版本估算；公开 MVP 更合理的区间为 22～32 人日。

### 更新后的迁移策略

| Decision | Rationale |
|----------|-----------|
| 按固定源 commit 选择性抽取 | 便于追踪来源，降低商业代码误发布和无限追随源分支的风险。 |
| 使用垂直切片迁移 | 每个阶段同时验证后端、前端、数据和测试，减少后期集中返工。 |
| 创建 Starter baseline migration | Starter 无需承担 Enterprise 数据表和历史升级路径。 |
| 默认通过公开交换格式迁移用户数据 | 避免两个项目在数据库和本地文件路径层形成隐式依赖。 |
| 将公开审计设为 release gate | 防止敏感信息、私有资源和不适合公开分发的依赖进入仓库。 |
| 将引流体验设为产品验收项 | Starter 的成功标准不仅是运行成功，还包括用户能理解 Enterprise 的后续价值。 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 目标目录为空，没有现成项目约束 | 先写入迁移计划、发现记录和进度记录；后续 Phase 2 再初始化项目结构。 |
| 原计划的阶段粒度过大 | 重排为基础设施、数据集/标注、训练、模型/导出、公开发布和 RC 验收。 |

## Resources

- Phase 1 功能矩阵：[`phase1_scope.md`](/Users/tony/PycharmProjects/YoloWebAgentStarter/phase1_scope.md)
- 逐层迁移清单：[`migration_matrix.md`](/Users/tony/PycharmProjects/YoloWebAgentStarter/migration_matrix.md)
- 固定源快照：[`source_snapshot.md`](/Users/tony/PycharmProjects/YoloWebAgentStarter/source_snapshot.md)
- 总迁移计划：[`task_plan.md`](/Users/tony/PycharmProjects/YoloWebAgentStarter/task_plan.md)
- 源项目说明：[`/Users/tony/PycharmProjects/YoloWebAgent/README.md`](/Users/tony/PycharmProjects/YoloWebAgent/README.md)
- 源项目总约束：[`/Users/tony/PycharmProjects/YoloWebAgent/AGENTS.md`](/Users/tony/PycharmProjects/YoloWebAgent/AGENTS.md)
- 后端约束：[`/Users/tony/PycharmProjects/YoloWebAgent/backend/AGENTS.md`](/Users/tony/PycharmProjects/YoloWebAgent/backend/AGENTS.md)
- 后端路由聚合：[`/Users/tony/PycharmProjects/YoloWebAgent/backend/app/api/routes.py`](/Users/tony/PycharmProjects/YoloWebAgent/backend/app/api/routes.py)
- 后端启动入口：[`/Users/tony/PycharmProjects/YoloWebAgent/backend/app/main.py`](/Users/tony/PycharmProjects/YoloWebAgent/backend/app/main.py)
- 数据库模型：[`/Users/tony/PycharmProjects/YoloWebAgent/backend/app/core/models.py`](/Users/tony/PycharmProjects/YoloWebAgent/backend/app/core/models.py)
- 模型路由：[`/Users/tony/PycharmProjects/YoloWebAgent/backend/app/api/routers/models.py`](/Users/tony/PycharmProjects/YoloWebAgent/backend/app/api/routers/models.py)

## Visual/Browser Findings

- 本轮没有使用浏览器或图像工具。
