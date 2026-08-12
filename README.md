# YoloWebAgentStarter

面向本地单用户的 YOLO 数据集工作台：完成图片/视频帧导入、人工标注、数据校验、YOLO/COCO 交换、本地训练、评估和受管模型产物的一条轻量闭环。

```text
图片 → 数据集 → 标注 → 校验 → YOLO 导入 / 导出 → 本地训练 → PT / ONNX
```

> **发布状态：阻断。** 本仓库基于固定的 YoloWebAgent 快照进行选择性派生，但该上游树尚无可核验的顶层许可证或 NOTICE。仓库中的 MIT 文件不能单方面完成再许可；在取得权利人书面授权或可核验上游许可证前，**不得发布公开 release 或宣称为已获授权的开源衍生版本**。请先阅读[来源授权门槛](docs/provenance/UPSTREAM_AUTHORIZATION.md)、[逐模块来源审计](migration_matrix.md)和[source snapshot](source_snapshot.md)。

## 为什么使用它

- 在本机完成 detect、segment、OBB 和 classify 数据集的标注与训练准备。
- 用持久化的 `train` / `val` / `test` split 贯通校验、YOLO 导出和本地训练。
- 提供 SAM 的 segment 交互建议；所有建议都必须由用户确认后按普通 polygon 标注保存。
- 训练结果只以受管的 `best.pt`、`last.pt` 和静态 FP32 ONNX 产物形式进入模型库。
- 默认只监听 `127.0.0.1`，数据、数据库、导出和训练文件均保留在本机。

## 功能一览

| 能力 | 说明 |
|---|---|
| 数据集 | 创建数据集和类别、浏览器上传、视频抽帧、受限本地目录扫描、持久化 split 管理、批量及可复现自动 split、只读重复/相似图报告和派生切片数据集 |
| 标注 | detect bbox、segment polygon、OBB 选择/移动/缩放/旋转、单标签 classify |
| SAM | segment 的框选/点选建议；未配置模型时仅提供明确标识的 review-only 框形建议 |
| 数据交换 | YOLO detect / segment / OBB ZIP 导入与导出；YOLO classify 目录布局导入与导出；detect/segment COCO ZIP 导入与导出 |
| 训练 | 本地 FIFO 队列、日志、进度、停止控制、从受管 last.pt 续跑、指标摘要/趋势、配置快照和 best/last checkpoint |
| 模型 | 训练产物的受管 PT 下载、持久化图片快速测试、同数据集模型比较、可审阅预标注、本地 split 评估/错误样本、去重 FP32 ONNX 导出 |
| 数据质量 | 标注覆盖率、类别分布、小目标、重叠 bbox 和类别失衡提示 |
| 安全边界 | 受管存储根目录、导入目录边界、ZIP 防资源耗尽限制、默认 localhost 绑定 |

### 任务支持

| 任务 | 标注表示 | YOLO 交换 | 本地训练 |
|---|---|---|---|
| `detect` | bbox | 支持 | 支持 |
| `segment` | polygon；可选 SAM 建议 | 支持 | 支持 |
| `obb` | 绝对像素的中心、尺寸、角度 | 支持 | 支持 |
| `classify` | 每张图片一个类别 | 标准 `split/class/image` 目录 | 支持 |

不包含：登录/RBAC、协作、Agent、Workflow、无人值守批量自动标注、文本提示分割、Deployment、pose、云训练和分布式训练。完整边界见 [Community v2 功能矩阵](phase1_scope.md)。

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

训练完成后，`best.pt` 与 `last.pt` 会登记到受管模型目录；在“模型”工作区可下载 PT、编辑元数据、归档，或生成 FP32 ONNX。

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
| NVIDIA CUDA | `0` 或 API 的设备值 | 支持传递给 Ultralytics；需要匹配的 CUDA PyTorch、驱动和硬件 |

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

不可信 YOLO ZIP 在写入前会检查最多 2,000 个成员、单成员 100 MiB、总解压量 250 MiB 和 100:1 压缩比；图片逐成员流式写入受管目录。目录扫描会拒绝导入根目录外路径和逃逸的软链接。

## 安全与运行边界

这是本地单用户软件，不提供认证、授权、TLS 或多租户隔离。请不要将服务绑定到 `0.0.0.0`、转发端口或直接放到公网反向代理之后。

安全报告流程目前也是公开发布的阻断项；请阅读 [SECURITY.md](SECURITY.md)。发布前的全部检查见 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。

## 项目结构

```text
backend/app/       FastAPI、领域服务、SQLite/Alembic 和本地训练队列
frontend/src/      React 标注工作台、训练与模型管理界面
scripts/           微型数据集、CPU 烟雾测试和发布门槛脚本
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

CPU 冒烟会实际运行 detect、segment、OBB、classify 的一轮微型训练，并校验 detect ONNX 导出。详见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CHANGELOG.md](CHANGELOG.md)。

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
