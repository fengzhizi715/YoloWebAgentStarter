# YoloWebAgentStarter Community v2 功能矩阵

## 1. 定位

YoloWebAgentStarter Community v2 是一个本地运行的 YOLO 数据集与训练工作台，用于完成：

```text
图片 → 数据集 → detect / segment / OBB / classify 人工标注 → 校验 → 本地训练 → PT → ONNX
```

Starter 是独立开源引流产品，不依赖 Enterprise，也不承诺包含完整训练闭环、企业协作或部署交付能力。

## 2. Phase 1 锁定决策

| 决策项 | Community v2 决策 | 理由 |
|--------|---------|------|
| TaskType | detect、segment、obb、classify | 复用固定 YoloWebAgent 快照的成熟任务契约，保持独立运行时 |
| OBB | included | 使用绝对像素中心、尺寸、角度持久化与 YOLO 八点标签 |
| classify | included | 每张图一个类别，使用 YOLO 分类目录布局 |
| pose | excluded | 当前 Enterprise 闭环尚未完全收口 |
| 数据交换 | YOLO import / export | 公开、直观、与训练链路直接相关 |
| native archive | excluded | 不作为首版数据库迁移或项目交换格式 |
| COCO exchange | detect / segment ZIP 导入与导出 | 只支持标准 bbox / polygon 表示 |
| 基础校验 | included | 训练前必须验证标注、类别和坐标合法性 |
| 基础质量报告 | included | 覆盖率、类别分布、小目标、重叠 bbox 和类别失衡；不生成派生数据集 |
| 高级数据准备 | bounded | 视频抽帧、只读重复/相似图、detect/segment 派生切片；无后台工作流和批量清理 |
| 训练方式 | 本地 Ultralytics | 不提供云训练和远程分布式调度 |
| 训练设备 | CPU 必测；CUDA 单 GPU / 本地多 GPU DDP 与 MPS best-effort | CI 以 CPU 为基线，硬件能力按本机可用性启用 |
| 模型来源 | Starter 训练产物 | 首版不提供任意外部 PT 导入中心 |
| PT 能力 | best.pt / last.pt 管理与下载 | PT 不需要格式转换 |
| ONNX | FP32 导出 | 排除 FP16、INT8、TensorRT、OpenVINO |
| 模型快速测试 | included | 受管 PT 对临时上传图片本机推理，结果不自动写回数据集 |
| 用户与权限 | 无登录、本地单用户 | Starter 不迁移 Auth / RBAC / License |
| 默认网络边界 | `127.0.0.1` | 无认证模式不得默认暴露公网 |
| 首发平台 | macOS、Linux | Windows 暂不承诺正式支持 |
| 维护策略 | 基础修复维护 | 优先处理安装阻断、严重缺陷和安全问题，不承诺高频功能迭代 |

## 3. 功能矩阵

### 3.1 数据集与标注

| 功能 | Community v2 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 创建、查看、修改、删除数据集 | included | included | 本地 SQLite 和文件目录一致 |
| 图片目录扫描导入 | included | included | 识别支持格式并记录尺寸 |
| 视频导入与抽帧 | included | included | mp4 / mov / avi 本地抽帧，最多 1,000 帧 |
| 类别创建与查看 | included | included | class_index 稳定且不越界 |
| detect bbox 标注 | included | included | 创建、拖拽、缩放、保存 |
| segment polygon 标注 | included | included | 绘制、编辑、保存，至少 3 点 |
| OBB / classify | included | included | Starter API、UI、schema、YOLO 交换与训练闭环可用 |
| pose | excluded | included | 不在 Community v2 范围 |
| SAM 交互建议 | included | included | 仅 segment 框/点提示，多边形须人工确认保存 |
| 数据集级自动标注 | included | included | 从数据集卡片启动本地受管 PT 任务，支持显式类别映射、置信度/IoU、二次确认清理旧标注、进度、日志和取消；与训练在本机互斥，结果必须人工审核 |
| SAM 设置 | included | included | 本地设置页管理启用状态、模型、设备、推理尺寸和无模型回退行为 |
| 基础数据集校验 | included | included | 坐标、类别、空标注、格式问题可返回 |
| 高级质量报告 | included | included | 覆盖率、类别分布、小目标、重复 bbox、重复/相似图片；不含 Agent 诊断 |
| split 持久化 | included | included | 导出和训练复用已有 split |
| YOLO 导入 | included | included | detect / segment / OBB 标签和 classify 目录均可导入 |
| YOLO 导出 | included | included | 对应任务可训练数据结构正确 |
| native archive | excluded | included | Starter 不暴露 native archive 端点 |
| COCO detect / segment 数据交换 | included | included | 可导入和导出 ZIP；不支持 OBB / classify |

### 3.2 YOLO 训练

| 功能 | Community v2 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 训练默认值与预览 | included | included | 输出解析后的最终配置和风险提示 |
| 本地训练任务创建 | included | included | detect / segment / OBB / classify 权重族匹配 |
| 单机任务队列 | included | included | 同时只运行允许数量的任务 |
| 本地多 GPU 训练 | included | included | 选择两张或以上 CUDA GPU，按 `device=0,1` 交给 Ultralytics DDP；不含远程调度 |
| 任务状态、日志、摘要 | included | included | 前端可持续查看 |
| 停止训练 | included | included | 进程和状态可收敛 |
| pause / unpause | excluded | included | Starter 不依赖平台信号语义 |
| checkpoint 列表 | included | included | 可查看 last / best 和可恢复 checkpoint |
| 失败恢复 / checkpoint 续跑 | included | included | 从 Starter 受管 `last.pt` 创建新任务；不接受任意外部 checkpoint |
| 独立 Evaluation | included | included | 对保存 split 同步运行本地评估并持久化错误样本；不含后台任务或 Workflow |
| Workflow / Agent | excluded | included | 不触发商业自动化闭环 |
| 云训练 / 远程分布式调度 | excluded | excluded/future | 本地多 GPU DDP 已纳入，但不提供云端 worker 或远程调度 |

### 3.3 模型管理与导出

| 功能 | Starter v1 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| best.pt / last.pt 自动入库 | included | included | 训练完成后形成 ModelVersion |
| 模型列表与详情 | included | included | 显示来源、TaskType、训练任务和基础指标 |
| 备注、归档、恢复、删除 | included | included | 数据库状态一致 |
| PT 下载 | included | included | 只访问受管模型文件 |
| ONNX FP32 导出 | included | included | 从 PT 生成有效 `.onnx` 并可下载 |
| 外部 PT 导入 | excluded | included | Community v2 不提供导入中心 |
| 模型比较 | included | included | 仅限同数据集、同任务模型的基础指标对比 |
| 模型快速测试 | included | included | 受管 PT 可上传图片测试，支持 detect / segment / OBB / classify；结果和输入图持久化到受管模型目录 |
| 独立 Evaluation | included | included | 四任务复用已保存 split 和上游 YOLO val，持久化任务原生指标、图表与最多 200 个错误样本 |
| TensorRT / OpenVINO / FP16 / INT8 | excluded | included | 无部署 runtime 和量化依赖 |
| Benchmark / 部署包 / 内网下发 | excluded | included | 无相关表、API 和页面 |

### 3.4 产品与运维

| 功能 | Community v2 | Enterprise | 验收标准 |
|------|------------|------------|----------|
| 本地单用户 | included | supported | 默认只监听 localhost |
| 登录 / RBAC / 用户管理 | excluded | included | Starter 无 AuthProvider 和用户表 |
| 商业 License | excluded | included | Starter 无 offline_license 依赖 |
| LLM / Agent / Workflow | excluded | included | Starter 无配置、路由和页面 |
| Runtime logs 管理页 | included | included | 读取 Starter 数据目录中的本地后端日志，支持行数、级别和内容筛选 |
| SQLite | included | included | Starter 唯一正式数据库 |
| MySQL | excluded | included | Starter 不迁移 PyMySQL 和迁移指南 |
| i18n | included | included | 只保留 Starter 使用的中英文消息 |
| CI | included | included | backend test、frontend test/build、CPU smoke |

## 4. 发布成功标准

Community v2 必须在全新环境中完成：

1. 创建数据集并扫描图片目录。
2. 创建类别并完成 bbox 或 polygon 标注。
3. 校验并导出任一支持的 YOLO detect / segment / OBB / classify 数据集。
4. 使用支持的基础权重发起本地训练。
5. 查看任务状态、日志和训练摘要。
6. 获得并管理 best.pt / last.pt。
7. 下载 PT，并从 PT 导出和下载 ONNX FP32。
8. 在 segment 数据集上配置 SAM（可选）并确认交互式建议后保存 polygon 标注。
9. 在数据集卡片选择匹配的受管 PT 运行自动标注，审核 `auto` 来源结果后再训练。
10. 全流程不读取 Enterprise 数据库、虚拟环境或文件目录。

## 5. 变更规则

- Phase 2 开始后新增能力必须显式修改本矩阵，并同步调整工作量、测试和发布门槛。
- native archive 和 Windows 正式支持属于后续候选，不得在迁移中隐式带入。
- 如果某个排除功能被核心代码直接 import，应重写依赖边界，而不是把整个商业模块复制进 Starter。
