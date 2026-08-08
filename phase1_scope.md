# YoloWebAgentStarter v1 功能矩阵

## 1. 定位

YoloWebAgentStarter v1 是一个本地运行的 YOLO 数据集与训练工作台，用于完成：

```text
图片 → 数据集 → detect / segment 人工标注 → 校验 → 本地训练 → PT → ONNX
```

Starter 是独立开源引流产品，不依赖 Enterprise，也不承诺包含完整训练闭环、企业协作或部署交付能力。

## 2. Phase 1 锁定决策

| 决策项 | v1 决策 | 理由 |
|--------|---------|------|
| TaskType | detect、segment | 覆盖最主流人工标注和训练场景，控制首版测试矩阵 |
| OBB | excluded | 推迟到后续版本，不进入首发维护范围 |
| classify | excluded | 目录型数据结构和训练流程不同，推迟到后续版本 |
| pose | excluded | 当前 Enterprise 闭环尚未完全收口 |
| 数据交换 | YOLO import / export | 公开、直观、与训练链路直接相关 |
| native archive | excluded | 不作为首版数据库迁移或项目交换格式 |
| COCO export | excluded | 不属于首版核心链路 |
| 基础校验 | included | 训练前必须验证标注、类别和坐标合法性 |
| 基础质量报告 | excluded | 保留 validation，排除诊断型 quality report |
| 高级数据准备 | excluded | 视频抽帧、tiling、去重、相似图和派生数据集均不纳入 |
| 训练方式 | 本地 Ultralytics | 不提供云训练和分布式调度 |
| 训练设备 | CPU 必测；CUDA / MPS best-effort | CI 以 CPU 为基线，硬件能力按本机可用性启用 |
| 模型来源 | Starter 训练产物 | 首版不提供任意外部 PT 导入中心 |
| PT 能力 | best.pt / last.pt 管理与下载 | PT 不需要格式转换 |
| ONNX | FP32 导出 | 排除 FP16、INT8、TensorRT、OpenVINO |
| 模型快速测试 | excluded | 不进入首版模型管理范围 |
| 用户与权限 | 无登录、本地单用户 | Starter 不迁移 Auth / RBAC / License |
| 默认网络边界 | `127.0.0.1` | 无认证模式不得默认暴露公网 |
| 首发平台 | macOS、Linux | Windows 暂不承诺正式支持 |
| 维护策略 | 基础修复维护 | 优先处理安装阻断、严重缺陷和安全问题，不承诺高频功能迭代 |

## 3. 功能矩阵

### 3.1 数据集与标注

| 功能 | Starter v1 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 创建、查看、修改、删除数据集 | included | included | 本地 SQLite 和文件目录一致 |
| 图片目录扫描导入 | included | included | 识别支持格式并记录尺寸 |
| 视频导入与抽帧 | excluded | included | Starter 无入口和依赖 |
| 类别创建与查看 | included | included | class_index 稳定且不越界 |
| detect bbox 标注 | included | included | 创建、拖拽、缩放、保存 |
| segment polygon 标注 | included | included | 绘制、编辑、保存，至少 3 点 |
| OBB / classify / pose | excluded | included | Starter API、UI、schema 不暴露 |
| SAM / AI 自动标注 | excluded | included | Starter 无模型配置和预测表 |
| 基础数据集校验 | included | included | 坐标、类别、空标注、格式问题可返回 |
| 高级质量报告 | excluded | included | Starter 不迁移 quality service |
| split 持久化 | included | included | 导出和训练复用已有 split |
| YOLO 导入 | included | included | detect / segment 数据可导入 |
| YOLO 导出 | included | included | detect / segment 可训练数据结构正确 |
| native / COCO 数据交换 | excluded | included | Starter 不暴露相关端点 |

### 3.2 YOLO 训练

| 功能 | Starter v1 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 训练默认值与预览 | included | included | 输出解析后的最终配置和风险提示 |
| 本地训练任务创建 | included | included | detect / segment 权重族匹配 |
| 单机任务队列 | included | included | 同时只运行允许数量的任务 |
| 任务状态、日志、摘要 | included | included | 前端可持续查看 |
| 停止训练 | included | included | 进程和状态可收敛 |
| pause / unpause | excluded | included | Starter 不依赖平台信号语义 |
| checkpoint 列表 | included | included | 可查看 last / best 和可恢复 checkpoint |
| 失败恢复 | included | included | 启动后清理失活任务并可重新发起 |
| 自动评估 / Workflow / Agent | excluded | included | 训练完成后不触发商业闭环 |
| 云训练 / 分布式训练 | excluded | excluded/future | 不在 Starter 范围 |

### 3.3 模型管理与导出

| 功能 | Starter v1 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| best.pt / last.pt 自动入库 | included | included | 训练完成后形成 ModelVersion |
| 模型列表与详情 | included | included | 显示来源、TaskType、训练任务和基础指标 |
| 备注、归档、恢复、删除 | included | included | 数据库状态一致 |
| PT 下载 | included | included | 只访问受管模型文件 |
| ONNX FP32 导出 | included | included | 从 PT 生成有效 `.onnx` 并可下载 |
| 外部 PT 导入 | excluded | included | Starter v1 不提供导入中心 |
| 模型比较 | excluded | included | 无 compare API 和 UI |
| 模型快速测试 | excluded | included | 无 test / save-test API |
| 独立 Evaluation | excluded | included | 无评估任务和错误样本 |
| TensorRT / OpenVINO / FP16 / INT8 | excluded | included | 无部署 runtime 和量化依赖 |
| Benchmark / 部署包 / 内网下发 | excluded | included | 无相关表、API 和页面 |

### 3.4 产品与运维

| 功能 | Starter v1 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 本地单用户 | included | supported | 默认只监听 localhost |
| 登录 / RBAC / 用户管理 | excluded | included | Starter 无 AuthProvider 和用户表 |
| 商业 License | excluded | included | Starter 无 offline_license 依赖 |
| LLM / Agent / Workflow | excluded | included | Starter 无配置、路由和页面 |
| Runtime logs 管理页 | excluded | included | 仅保留标准进程日志 |
| SQLite | included | included | Starter 唯一正式数据库 |
| MySQL | excluded | included | Starter 不迁移 PyMySQL 和迁移指南 |
| i18n | included | included | 只保留 Starter 使用的中英文消息 |
| CI | included | included | backend test、frontend test/build、CPU smoke |

## 4. 发布成功标准

Starter v1 必须在全新环境中完成：

1. 创建数据集并扫描图片目录。
2. 创建类别并完成 bbox 或 polygon 标注。
3. 校验并导出 YOLO detect / segment 数据集。
4. 使用支持的基础权重发起本地训练。
5. 查看任务状态、日志和训练摘要。
6. 获得并管理 best.pt / last.pt。
7. 下载 PT，并从 PT 导出和下载 ONNX FP32。
8. 全流程不读取 Enterprise 数据库、虚拟环境或文件目录。

## 5. 变更规则

- Phase 2 开始后新增能力必须显式修改本矩阵，并同步调整工作量、测试和发布门槛。
- OBB、classify、native archive、模型快速测试和 Windows 正式支持属于后续候选，不得在迁移中隐式带入。
- 如果某个排除功能被核心代码直接 import，应重写依赖边界，而不是把整个商业模块复制进 Starter。
