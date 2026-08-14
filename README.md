# YoloWebAgentStarter

面向本地单用户的 YOLO 数据集工作台：完成图片/视频帧导入、人工标注、数据校验、YOLO/COCO 交换、本地训练、原生 YOLO 评估和受管模型产物的一条轻量闭环。

```text
图片 → 数据集 → 标注 → 校验 → YOLO 导入 / 导出 → 本地训练 → 受管 PT → split 评估 / FP32 ONNX
```

> **发布状态：阻断。** 本仓库基于固定的 YoloWebAgent 快照进行选择性派生，但该上游树尚无可核验的顶层许可证或 NOTICE。仓库中的 MIT 文件不能单方面完成再许可；在取得权利人书面授权或可核验上游许可证前，**不得发布公开 release 或宣称为已获授权的开源衍生版本**。请先阅读[来源授权门槛](docs/provenance/UPSTREAM_AUTHORIZATION.md)、[逐模块来源审计](migration_matrix.md)和[source snapshot](source_snapshot.md)。

## 为什么使用它

- 在本机完成 detect、segment、OBB 和 classify 数据集的标注与训练准备。
- 用持久化的 `train` / `val` / `test` split 贯通校验、YOLO 导出和本地训练。
- 提供 SAM 的 segment 交互建议；所有建议都必须由用户确认后按普通 polygon 标注保存。
- 在设置页管理 SAM 模型、推理设备、推理尺寸和无模型时的 review-only 回退策略。
- 训练页可选择 CPU、MPS、单张 CUDA GPU 或多张 CUDA GPU；本地多 GPU 使用 Ultralytics DDP。
- 提供本地后端运行日志页，支持尾部行数、级别和内容筛选；语言偏好保存在浏览器本地。
- 训练结果只以受管的 `best.pt`、`last.pt` 和静态 FP32 ONNX 产物形式进入模型库。
- 对受管 PT 后台运行上游同款 Ultralytics `val`，保存原生指标、日志、图表和最多 200 个可审阅错误样本。
- 默认只监听 `127.0.0.1`，数据、数据库、导出和训练文件均保留在本机。

## 功能一览

| 能力 | 说明 |
|---|---|
| 数据集 | 创建数据集和类别、浏览器上传、视频抽帧、受限本地目录扫描、持久化 split 管理、批量及可复现自动 split、只读重复/相似图报告和派生切片数据集 |
| 标注 | detect bbox、segment polygon、OBB 选择/移动/缩放/旋转、单标签 classify |
| SAM | segment 的框选/点选建议；未配置模型时仅提供明确标识的 review-only 框形建议 |
| 数据交换 | YOLO detect / segment / OBB ZIP 导入与导出；YOLO classify 目录布局导入与导出；detect/segment COCO ZIP 导入与导出 |
| 训练 | 本地 FIFO 队列、CPU/MPS/CUDA 单 GPU 或本地多 GPU DDP、日志、进度、停止控制、恢复中断任务或从受管 `last.pt` 创建继续训练任务、指标摘要/趋势、配置快照和 best/last checkpoint |
| 设置与日志 | SAM 设置、语言设置、本地运行日志查看与筛选 |
| 评估 | 后台原生 YOLO `val`、持久化 split、任务状态与恢复、日志、混淆矩阵、可用 PR 曲线和最多 200 个错误样本；segment 分别保留 box/mask 指标与曲线 |
| 模型 | 训练产物的受管 PT 下载、持久化图片快速测试、同数据集模型比较、可审阅预标注、去重 FP32 ONNX 导出 |
| 数据质量 | 标注覆盖率、类别分布、小目标、重叠 bbox 和类别失衡提示 |
| 安全边界 | 受管存储根目录、导入目录边界、ZIP 防资源耗尽限制、默认 localhost 绑定 |

### 任务支持

| 任务 | 标注表示 | YOLO 交换 | 本地训练 | split 评估 |
|---|---|---|---|---|
| `detect` | bbox | 支持 | 支持 | box P/R/mAP、图表和错误样本 |
| `segment` | polygon；可选 SAM 建议 | 支持 | 支持 | box/mask 两组 P/R/mAP、RLE 预测和图表 |
| `obb` | 绝对像素的中心、尺寸、角度 | 支持 | 支持 | OBB 指标、图表和 polygon IoU 错误样本 |
| `classify` | 每张图片一个类别 | 标准 `split/class/image` 目录 | 支持 | top-1 / top-5 指标 |

不包含：登录/RBAC、协作、Agent、Workflow、无人值守批量自动标注、文本提示分割、Deployment、pose、云训练和远程分布式调度。完整边界见 [Community v2 功能矩阵](phase1_scope.md)。

## 快速开始

### 依赖

- macOS 或 Linux
- Python 3.11 / 3.12
- Node.js 20+

从仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
npm --prefix frontend install
./run-backend.sh
```

在另一个终端启动前端：

```bash
./run-frontend.sh
```

打开 <http://127.0.0.1:5173>；API 地址为 <http://127.0.0.1:8000>。也可以执行 `./run-all.sh`（或 `sh run-all.sh`）同时启动两个服务，并以 `Ctrl+C` 一并停止。

> 首次以 `yolo11n.pt`、`yolo11n-seg.pt` 等命名权重训练时，Ultralytics 可能下载对应权重。请在确认适用许可后使用，且不要提交权重、数据集或产物。

## 首次使用流程

1. 创建 `detect`、`segment`、`obb` 或 `classify` 数据集。
2. 上传图片，或把图片放入 `data/imports/` 下的子目录并在界面中扫描。
3. 添加类别并标注：bbox、polygon、可旋转 OBB，或每图一个分类标签。
4. 确保 `train` 与 `val` 都至少有一张图片，运行数据集校验。
5. 导出/导入 YOLO 数据，或在训练页选择匹配的模型族并发起本地任务。
6. 训练完成后进入“模型”工作区，对受管 PT 选择 `train`、`val` 或 `test` split 创建后台评估任务。

训练完成后，`best.pt` 与 `last.pt` 会登记到受管模型目录；在“模型”工作区可下载 PT、编辑元数据、归档、生成 FP32 ONNX，或查看评估历史、原生指标、图表、日志和错误样本。评估与 YOLO 导出复用图片已经持久化的 split，不会重新随机切分。

更完整的操作步骤、SAM 说明和故障排查请见[五分钟上手](docs/quick-start.md)。可用以下命令生成可丢弃的微型 detect 数据集：

```bash
./.venv/bin/python scripts/create_tiny_demo.py /tmp/ywa-tiny-demo
```

## 设备支持

训练设备由 Ultralytics 在本机解析：

| 设备 | 在界面中的值 | 状态 |
|---|---|---|
| CPU | `cpu` 或 `auto` | 发布冒烟已覆盖 |
| Apple Silicon / Metal | `mps` | 支持传递给 Ultralytics；请在目标 Mac 上执行冒烟验证 |
| NVIDIA CUDA 单 GPU | `0`、`cuda:0` 或界面中的单 GPU | 支持传递给 Ultralytics；需要匹配的 CUDA PyTorch、驱动和硬件 |
| NVIDIA CUDA 多 GPU | `0,1` 或界面中的多 GPU | 使用 `device=0,1` 触发本地 Ultralytics DDP；不提供远程调度 |

本仓库当前的 CI 基线为 CPU。MPS 与 CUDA 的兼容性取决于目标主机的 PyTorch/Ultralytics 组合、驱动和可用硬件。

## 配置与数据位置

所有运行时数据默认位于被 Git 忽略的 `./data/`：SQLite 数据库、受管图片、训练任务、导出和模型文件。常用配置如下；完整示例见 [backend/.env.example](backend/.env.example)。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `YWA_DATA_DIR` | `./data` | Starter 受管运行时根目录 |
| `YWA_IMPORT_ROOT` | `./data/imports` | 服务端目录扫描允许访问的唯一根目录 |
| `YWA_HOST` / `YWA_PORT` | `127.0.0.1` / `8000` | 后端监听地址和端口 |
| `YWA_MAX_UPLOAD_MB` | `50` | 单次上传上限 |
| `YWA_MAX_YOLO_ARCHIVE_*` | 见示例文件 | YOLO ZIP 的数量、解压大小和压缩比限制 |
| `YWA_SAM_MODEL` | 未设置 | 启用真实 Ultralytics SAM 框/点提示的本地或命名检查点 |
| `YWA_SAM_DEVICE` | `auto` | SAM 设备请求，例如 `mps` 或 `cpu` |
| `YWA_SAM_IMGSZ` | `1024` | SAM 推理尺寸；设置页保存的值优先于环境默认值 |

设置页的 SAM 配置保存在 `YWA_DATA_DIR/settings.json`，语言偏好只保存在浏览器 localStorage；运行日志保存在 `YWA_DATA_DIR/logs/backend.log`，按 2 MiB 轮转并保留 3 个备份，日志页会合并展示保留文件中的最新行。

不可信 YOLO ZIP 在写入前会检查最多 2,000 个成员、单成员 100 MiB、总解压量 250 MiB 和 100:1 压缩比；图片逐成员流式写入受管目录。目录扫描会拒绝导入根目录外路径和逃逸的软链接。

## 安全与运行边界

这是本地单用户软件，不提供认证、授权、TLS 或多租户隔离。请不要将服务绑定到 `0.0.0.0`、转发端口或直接放到公网反向代理之后。

安全报告流程目前也是公开发布的阻断项；请阅读 [SECURITY.md](SECURITY.md)。发布前的全部检查见 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。

## 评估实现与产物

评估仅接受 Starter 训练任务登记的受管 Ultralytics PT 模型。HTTP 路由创建任务后立即返回；本地后台 runner 执行与固定上游相同的原生 `val` 合约：

```text
yolo <detect|segment|obb|classify> val ... plots=True save_json=True exist_ok=True
```

- detect 与 OBB 保存 box 指标；segment 同时解析 box 和 mask 两组 precision、recall、mAP50、mAP50-95；classify 保存 top-1、top-5。
- 产物通过受管 artifact API 提供，不允许请求评估目录外文件。通用任务使用 `PR_curve.png`；Ultralytics 8.3.40 的 segment 使用 `BoxPR_curve.png` 与 `MaskPR_curve.png`，页面会分别展示。
- `predictions.json` 按 Ultralytics 8.3.40 的真实格式分析：detect 使用左上角 `xywh`，segment 使用 pycocotools RLE mask，OBB 使用 `rbox` / `poly`。
- 错误样本包括 missed detection、false positive 和 low confidence；分析器使用已导出的同一 split 标签，最多持久化 200 条。classify 当前只展示原生指标，不生成目标级错误样本。
- 服务重启时，运行中的任务会标记失败并保留错误信息；尚未开始的任务会重新提交本地 runner。

图表只有在 Ultralytics 实际生成时才会出现。例如没有有效 TP 的极小随机模型可能不会生成 PR 曲线，但仍会产生指标、日志、混淆矩阵和预测 JSON。

## 上游对齐与独立运行

Community v2 的固定对齐基线是 YoloWebAgent commit `701f6e5a63b73f39e35f48fb6de7d2414401875a`。评估 runner、artifact manager、错误样本分析和详情面板沿用上游模块边界，并只裁剪到 detect、segment、OBB、classify；Ultralytics 8.3.40 文件名与 JSON 适配是该上游合约的兼容扩展。

Starter 运行时不会 import、读取或依赖 YoloWebAgent/Enterprise 仓库，也不包含其 Auth、RBAC、License、Agent、Workflow、Evaluation 自动化回调、Deployment 或 pose 模块。逐文件关系见[迁移矩阵](migration_matrix.md)，固定快照与选择性抽取记录见[source snapshot](source_snapshot.md)。

## 项目结构

```text
backend/app/       FastAPI、领域服务、SQLite/Alembic、本地训练队列和评估 runner
frontend/src/      React 标注工作台、训练、模型管理和评估详情界面
scripts/           微型数据集、四任务 CPU 训练/segment-val 烟雾测试和发布门槛脚本
docs/              快速开始、依赖审计和来源授权材料
data/              默认运行时目录（忽略，不提交）
```

后端路由只适配 HTTP；数据集、标注、训练和文件访问均由领域服务与受管存储边界处理。Starter 不在运行时 import、读取或依赖上游/Enterprise 仓库。

## 开发与验证

请始终使用仓库的 `.venv`，并在提交前执行：

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
PYTHONPATH=backend .venv/bin/python scripts/run_cpu_smoke.py
```

标准测试会直接调用 Ultralytics 8.3.40 的 detect、segment、OBB validator 生成真实 JSON 合约，再交给错误样本分析器验证。CPU 冒烟会实际运行四任务的一轮微型训练，复用上游原生参数执行 segment `val(save_json=True, plots=True)`，并校验 box/mask 八项指标、pycocotools RLE、混淆矩阵、预测 JSON 和 detect ONNX 导出。整个冒烟使用临时目录且不保留模型或数据集；首次运行可能需要等待 Matplotlib 字体缓存和 ONNX 导出。

极小的离线随机模型不保证产生有效 TP，因此 CPU 冒烟不强制要求 PR 曲线存在；`BoxPR_curve.png` / `MaskPR_curve.png` 的 Ultralytics 8.3.40 命名契约由聚焦测试覆盖。详见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CHANGELOG.md](CHANGELOG.md)。

## 参与贡献

欢迎提交范围内的缺陷修复、测试、文档和本地单用户体验改进。提交前请：

1. 保持功能在 [Community v2 范围](phase1_scope.md) 内；新增任务类型、认证、云服务或 Enterprise 工作流请先讨论。
2. 为行为变化补充聚焦测试，并运行上述验证命令。
3. 不提交模型权重、数据集、客户材料、凭据、构建日志或绝对路径。
4. 保留并更新来源记录；不得引入上游/Enterprise 的运行时依赖。

完整规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证、来源与第三方组件

仓库包含 MIT `LICENSE`，但其并不单独授予固定 YoloWebAgent 快照派生部分的再许可权；在上游授权门槛通过前，本项目不具备可公开发布条件。不要因为仓库中存在 MIT 文件而推断整个代码库已获授权。

- [上游授权发布门槛](docs/provenance/UPSTREAM_AUTHORIZATION.md)
- [固定来源快照](source_snapshot.md)
- [逐模块迁移审计](migration_matrix.md)
- [第三方依赖与模型权重说明](THIRD_PARTY_NOTICES.md)

Ultralytics 和任何 SAM/YOLO 权重均遵循各自的许可证与分发条款；发布者必须针对实际发布方式单独复核。
