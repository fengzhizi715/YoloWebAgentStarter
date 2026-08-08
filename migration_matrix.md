# YoloWebAgentStarter v1 迁移矩阵

## 1. 状态定义

| 状态 | 含义 |
|------|------|
| keep_review | 可复用主体实现，但必须检查命名、路径、依赖和公开内容 |
| rewrite | 复用业务意图或局部算法，重新建立 Starter 边界 |
| exclude | 不进入 Starter v1，且不能作为隐式依赖 |
| defer | 首版不迁移，未来重新评估 |

所有路径均相对于已归档的源项目，固定基线见 `source_snapshot.md`。

## 2. 后端模块矩阵

### 2.1 应用入口与 Core

| 源模块 | 状态 | Starter 落点 / 处理方式 |
|--------|------|-------------------------|
| `backend/app/main.py` | rewrite | 新建本地单用户启动入口；移除 Auth、License、SAM、Deployment capability 和商业任务恢复 |
| `backend/app/api/routes.py` | rewrite | 只聚合 system、datasets、training、models |
| `core/app_config.py`、`env.py`、`errors.py`、`ids.py`、`logging.py`、`task_types.py`、`time.py` | keep_review | 保留通用能力，删除商业配置和无用 TaskType |
| `core/paths.py`、`storage.py`、`file_utils.py` | rewrite | 使用 Starter 独立数据根目录并强化目录边界 |
| `core/database.py` | rewrite | SQLite-only；移除 MySQL 和旧库 patcher 负担 |
| `core/models.py` | rewrite | 仅定义 Starter 核心表 |
| `core/schemas.py` | rewrite | 只保留 detect/segment、dataset、image、class 和 YOLO schema |
| `core/idempotency.py` | keep_review | 仅在训练/导出任务实际使用时迁移 |
| `core/system.py` | rewrite | 只暴露 Starter 版本、设备和基础运行环境 |
| `core/alembic_utils.py` | exclude | Starter 使用全新 baseline migration |
| `core/annotation_sources.py` | rewrite | 只保留 manual / imported |
| `core/diagnostic_config.py`、`core/sam_config.py` | exclude | 属于诊断和 SAM 配置 |
| `core/license/` | exclude | Starter 无商业授权 |
| `backend/app/auth/` | exclude | Starter 无登录和 RBAC |

### 2.2 数据集与标注

| 源模块 | 状态 | Starter 落点 / 处理方式 |
|--------|------|-------------------------|
| `dataset/service.py` | rewrite | 移除 pose、created_by、preparation 和商业关联 |
| `dataset/scanner.py` | keep_review | 保留图片扫描、格式和尺寸读取 |
| `dataset/yolo_split_dirs.py` | keep_review | 保留 split 目录识别 |
| `dataset/annotation/converters.py` | rewrite | 只保留 bbox / polygon 成对转换 |
| `dataset/annotation/families.py` | rewrite | 只允许 detect / segment，混合 family 明确报错 |
| `dataset/annotation/service.py` | rewrite | 移除 pose、AI、auto 和用户上下文 |
| `dataset/annotation/ai/` | exclude | AI 辅助标注不进入 v1 |
| `dataset/annotation/auto/` | exclude | 自动标注和 YOLO result parser 不进入 v1 |
| `dataset/validation/` | rewrite | 保留 bbox、polygon、class_index、空标注等基础规则 |
| `dataset/exchange/yolo/` | rewrite | 只保留 detect / segment import/export、split 和 data.yaml |
| `dataset/exchange/splits.py` | keep_review | 复用已有 split，不重新随机覆盖 |
| `dataset/exchange/archive.py`、`native_*`、`import_service.py` | exclude | native archive 不进入 v1 |
| `dataset/exchange/coco_exporter.py` | exclude | COCO 不进入 v1 |
| `dataset/exchange/validators.py`、`schemas.py` | rewrite | 仅保留 YOLO 交换所需部分 |
| `dataset/ingestion/` | exclude | 用简单目录扫描代替 Import Center 任务系统 |
| `dataset/preparation/` | exclude | 高级数据准备不进入 v1 |
| `dataset/quality/` | exclude | 高级质量报告不进入 v1 |
| `backend/app/pose/` | exclude | pose 不进入 v1 |

### 2.3 训练

| 源模块 | 状态 | Starter 落点 / 处理方式 |
|--------|------|-------------------------|
| `training/config/` | rewrite | 保留 detect/segment 配置、权重族、batch、device、imgsz、epochs |
| `training/application/dataset_summary.py` | rewrite | 只读取 Starter dataset / split / annotation |
| `training/application/dataset_training_service.py` | rewrite | 移除 Auth、Workflow 和商业任务关联 |
| `training/application/job_service.py` | rewrite | 保留本地队列和重启恢复，移除商业回调 |
| `training/runtime/active_process.py` | keep_review | 保留本地进程注册和停止 |
| `training/runtime/device_service.py` | keep_review | CPU 基线，CUDA/MPS best-effort |
| `training/runtime/runner.py` | rewrite | 移除 Workflow、Evaluation、Iteration；只运行训练和产物入库 |
| `training/artifacts/` | keep_review | 保留任务路径、checkpoint 和模型产物 |
| `training/observability/log_store.py`、`metrics_parser.py`、`summary.py` | keep_review | 保留本地日志和基础训练指标 |
| `training/observability/telemetry_service.py` | exclude | Starter 不引入隐式外发遥测 |
| `training/schemas.py`、`job_schemas.py`、`errors.py` | rewrite | 收缩到 Starter API |
| `training/service.py`、`center_service.py` | rewrite | 保留稳定服务入口，移除权限和商业动作 |
| training 根目录兼容 wrapper | defer | 仅在迁移调用方确实需要时保留薄入口 |

### 2.4 YOLO 引擎、模型与导出

| 源模块 | 状态 | Starter 落点 / 处理方式 |
|--------|------|-------------------------|
| `yolo/cli_executable.py` | keep_review | 保留虚拟环境内 CLI 解析 |
| `yolo/engines/base.py` | rewrite | 只定义 train/load/export_onnx 所需接口 |
| `yolo/engines/factory.py` | keep_review | 仅注册 Ultralytics |
| `yolo/engines/ultralytics.py` | rewrite | 删除 Deployment schema、FP16/INT8/OpenVINO/TensorRT 分支 |
| `models/schemas.py` | rewrite | 保留基础模型版本和 ONNX 响应 |
| `models/service.py` | rewrite | 移除 Auth、自动标注 parser、模型测试和商业指标继承 |
| `models/export_metrics.py` | exclude | 不继承商业 deployment evaluation 指标 |
| `deployment/` | exclude | 不迁移完整部署领域 |
| 最小 ONNX exporter | rewrite | 从 Ultralytics engine 中抽取 FP32 ONNX 导出并在 models 领域调用 |

### 2.5 完整排除的后端领域

| 领域 | 状态 |
|------|------|
| `active_learning/` | exclude |
| `agent/` | exclude |
| `ai_annotation/` | exclude |
| `auto_annotation/` | exclude |
| `deployment/`（除重写后的最小 ONNX 逻辑） | exclude |
| `evaluation/` | exclude |
| `import_center/` | exclude |
| `iteration/` | exclude |
| `logs/` runtime 管理 API | exclude |
| `providers/`、`vision/`、`vision_language/` | exclude |
| `sam/` | exclude |
| `settings/` 商业设置 | exclude |

## 3. API 迁移矩阵

Starter 不承诺与 Enterprise API 完全兼容；以下端点按 Starter 语义重建。

### 3.1 保留/重写

| API 组 | 状态 | Starter 目标 |
|--------|------|--------------|
| `GET /system/info` | rewrite | Starter 名称、版本、设备和数据目录状态 |
| Dataset CRUD | rewrite | `/datasets`、`/datasets/{id}` |
| 图片扫描 | rewrite | `/datasets/{id}/scan-images`，限制本地目录边界 |
| 类别 | rewrite | `/datasets/{id}/classes` |
| 图片与标注 | rewrite | `/datasets/{id}/images`、`/images/{id}`、文件和 annotations |
| 数据集校验 | rewrite | `/datasets/{id}/validate` |
| YOLO import/export | rewrite | 只支持 detect / segment |
| Training defaults/preview | rewrite | 只返回 Starter 参数 |
| Training jobs | rewrite | 统一使用 `/training/jobs` 和 `/training/jobs/{id}` |
| Training logs/summary/stop/delete | rewrite | 移除 pause/unpause/resume 旧端点 |
| Training profiles | rewrite | 保留基础配置模板 |
| Model list/detail/update/archive/restore/delete | rewrite | 只处理 Starter ModelVersion |
| Model PT download | rewrite | 文件必须位于受管模型目录 |
| `POST /models/{id}/export-onnx` | rewrite | 仅 FP32 ONNX |

### 3.2 排除

| API 组 | 状态 |
|--------|------|
| `/license/*`、`/auth/*`、`/users/*` | exclude |
| `/agent/*`、`/workflows/*`、`/workflow-runs/*`、`/iterations/*` | exclude |
| `/sam/*`、`/vision/*`、`/ai/*`、`/ai-annotation/*` | exclude |
| `/evaluation/*`、`/active-learning/*`、`/auto-annotation/*` | exclude |
| `/preparation/*`、`/import/jobs/*` | exclude |
| native archive、COCO export、quality report | exclude |
| model compare、model test、save-test | exclude |
| `/deployment/*`、deployment packages、benchmarks、reports | exclude |
| runtime logs 管理端点 | exclude |

## 4. 数据库迁移矩阵

Starter 创建全新 baseline migration，不复制现有迁移历史。

| Enterprise 表 | Starter 决策 | 处理方式 |
|---------------|--------------|----------|
| `datasets` | rewrite | 删除 pose schema、preparation 和商业关联字段 |
| `class_labels` | keep_review | 保留 dataset、name、class_index |
| `image_items` | rewrite | 保留文件、尺寸、split 和基础来源；删除高级 lineage 字段 |
| `annotations` | rewrite | 只允许 bbox / polygon，source 只保留 manual / imported |
| `training_tasks` | rewrite | 删除 workflow、evaluation、created_by 和商业闭环字段 |
| `training_profiles` | rewrite | 只保留 Starter 支持参数 |
| `model_versions` | rewrite | 删除 created_by、部署指标继承和商业关联 |
| `import_jobs` | exclude | Starter 使用同步目录扫描 |
| `users`、`permissions`、`role_permissions`、`audit_logs` | exclude | 无 Auth / RBAC |
| `preparation_jobs`、`preparation_results` | exclude | 无高级数据准备 |
| Vision / AI / prediction / auto annotation 表 | exclude | 无 AI 标注 |
| Deployment / calibration / benchmark / report / package 表 | exclude | 无完整部署领域 |
| Evaluation / Active Learning / Iteration 表 | exclude | 无闭环诊断 |
| Agent chat / trace / Workflow 表 | exclude | 无 Agent / Workflow |

## 5. 前端迁移矩阵

### 5.1 页面

| 页面 | 状态 | 处理方式 |
|------|------|----------|
| `HomePage.tsx` | rewrite | 移除 Import Center、License、权限和商业快捷入口 |
| `AnnotationPage.tsx` | rewrite | 只保留 bbox / polygon，移除 OBB、Pose、SAM、AI Suggestions、权限和 License |
| `TrainingControlCenterPage.tsx` | rewrite | 移除 Workflow、Agent、权限；保留本地训练 |
| `TrainingTaskListPage.tsx` | rewrite | 保留创建、运行中和历史任务 |
| `TrainingTaskDetailPage.tsx` | rewrite | 移除 Agent 洞察和评估建议 |
| `TrainingProfilePage.tsx` | keep_review | 收缩 TaskType 和训练参数 |
| `ModelListPage.tsx` | rewrite | 移除比较、测试、Deployment Drawer 和 License |
| `ModelDetailPage.tsx` | rewrite | 只保留概览、训练来源、元数据、PT/ONNX |
| Login、Profile、Users、Settings、Logs | exclude | Starter 无 Auth 和商业设置 |
| Agent、Workflow、Evaluation、Deployment 页面 | exclude | Enterprise 能力 |
| Active Learning、Auto Annotation、Preparation、Quality 页面 | exclude | Enterprise 能力 |

### 5.2 组件与基础设施

| 目录/文件 | 状态 | 处理方式 |
|-----------|------|----------|
| `annotation/AnnotationCanvas.tsx` 等标注核心 | rewrite | 删除 OBB、Pose、SAM、prediction 分支 |
| `annotation/BBoxRect.tsx`、`PolygonShape.tsx` | keep_review | 保留 detect / segment 图形交互 |
| `annotation/annotationApi.ts`、`annotationTypes.ts` | rewrite | 移除 token、Auth、AI、SAM、OBB、Pose、classify 类型 |
| `components/ui/` | keep_review | 迁移实际被 Starter 页面引用的 primitives |
| `components/dataset-feature/` | keep_review | 移除 Agent/Workflow actions |
| `components/training-*` | rewrite | 删除商业诊断和不支持的任务控制 |
| `components/model-*` | rewrite | 只保留基础元数据和 PT/ONNX |
| Agent/workflow/deployment/evaluation/preparation/users 组件 | exclude | 不迁移 |
| `api/client.ts` | rewrite | 拆为 system、datasets、training、models client |
| `auth/`、`license/` | exclude | Starter 无认证授权 |
| `main.tsx`、`home/menuConfig.ts`、Topbar、Sidebar | rewrite | Starter 专用路由和导航 |
| locale 消息 | rewrite | 只迁移 Starter 实际使用 key |
| `types/index.ts` | rewrite | 按领域拆分，避免商业类型残留 |

## 6. 测试迁移矩阵

### 6.1 后端

| 测试类别 | 状态 |
|----------|------|
| app config、paths、errors、logging、time | keep_review |
| image scanning、annotation converters、segmentation polygon | rewrite |
| dataset validation、YOLO IO、dataset training jobs | rewrite |
| training、training center、command builder、Ultralytics engine、YOLO CLI | rewrite |
| model versions、PT download、ONNX FP32 export | rewrite |
| OBB、classify、pose、native archive、quality、Import Center | exclude |
| Auth、License、MySQL | exclude |
| Agent、Workflow、Evaluation、Deployment、Active Learning | exclude |
| Auto Annotation、SAM、VLM、Preparation、商业 release | exclude |

### 6.2 前端

| 测试类别 | 状态 |
|----------|------|
| annotation coordinate / polygon utilities | rewrite |
| API 基础请求和错误处理 | rewrite |
| UI primitives、dialog、pagination | keep_review |
| Starter 菜单、导航和路径 helper | rewrite |
| 训练日志/模型列表纯 helper | rewrite |
| Auth、License、Agent、Workflow、Deployment 等商业测试 | exclude |

## 7. 依赖迁移矩阵

### 7.1 Backend

| 依赖 | 状态 |
|------|------|
| FastAPI、Uvicorn、Pydantic | included |
| SQLAlchemy、Alembic | included |
| python-multipart、Pillow、PyYAML、NumPy | included |
| OpenCV | keep_if_used |
| Ultralytics | included |
| ONNX、ONNX Runtime、onnxscript、onnx_ir、ml_dtypes | verify_minimum |
| HTTPX | keep_if_used |
| PyMySQL | exclude |
| PyJWT、bcrypt | exclude |
| paramiko | exclude |
| offline_license wheel | exclude |
| EfficientSAM、OpenVINO、NNCF、TensorRT、PyCUDA | exclude |

### 7.2 Frontend

| 依赖 | 状态 |
|------|------|
| React、React DOM、TypeScript、Vite | included |
| Konva、react-konva | included |
| lucide-react | included |
| Vitest | included |
| `@xyflow/react` | exclude |
| AJV | exclude unless another retained feature proves it is needed |

## 8. Phase 2 执行顺序

1. 初始化干净 Git 和项目骨架。
2. 建立 Starter 数据目录、SQLite baseline 和最小 system API。
3. 按本矩阵迁移数据集/标注垂直切片。
4. 迁移训练垂直切片。
5. 迁移模型/PT/ONNX 垂直切片。
6. 每个切片完成后检查是否出现对 exclude 模块的 import。
7. 最后执行公开审计和引流体验建设。
