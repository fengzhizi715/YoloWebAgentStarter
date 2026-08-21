# 五分钟快速开始

## 1. 本地启动

请使用 Python 3.11 或 3.12，以及 Node.js 20 或更高版本。在仓库根目录执行：

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

打开 `http://127.0.0.1:5173`。两个开发服务默认仅绑定本机。也可以在仓库根目录执行 `./run-all.sh`（或 `sh run-all.sh`）同时启动两个进程；按 `Ctrl+C` 可同时停止它们。

## 2. 创建数据集并标注

1. 创建 `detect`、`segment`、`obb` 或 `classify` 数据集。
2. 上传图片，或将图片放入 `data/imports/` 下，再扫描相对目录。
3. 添加一个或多个类别，然后使用 bbox（detect）、polygon（segment）、旋转框（OBB）或每张图片一个类别（classify）进行标注。
4. 确保 `train` 和 `val` 中都至少有一张已标注图片，然后运行数据集校验。
5. detect、segment 和 OBB 使用 YOLO ZIP 文件导入/导出；分类任务使用标准的 `train/<class>/<image>` 目录布局。

对于 segment 数据集，请打开左侧边栏的 SAM 设置页面，配置模型、推理设备、图像尺寸和回退模式。相同的配置也可以通过 `YWA_SAM_*` 环境变量提供默认值。未配置模型时，界面会禁用点提示，并明确将框提示标记为仅供审核的矩形建议；系统不会声称执行了模型推理，也不会将该建议保存为 SAM 生成的标注。

对于 OBB 数据集，在画布空白区域拖动即可创建旋转框。点击已有 OBB 可选中它，拖动可移动，使用四个角点手柄可调整大小，使用顶部旋转手柄（或角度字段）可旋转。编辑器会在保存前将旋转框的角点限制在图片范围内。

如需生成一个可通过界面导入、用完即可丢弃的微型 detect 数据集：

```bash
./.venv/bin/python scripts/create_tiny_demo.py /tmp/ywa-tiny-demo
```

该命令会写入三个生成的 PNG 文件、标签和 `data.yaml`；不会下载或再分发模型权重。

## 3. 训练与导出

打开训练工作区，选择与数据集类型匹配的 YOLOv8、YOLO11 或 YOLO26 基础权重（例如 `yolov8n.pt`、`yolo11n-seg.pt` 或 `yolo26n-obb.pt`），然后选择 CPU、MPS 或 CUDA。存在多个 CUDA 设备时，可选择两个或更多 GPU，将 `device=0,1` 传递给本地 Ultralytics DDP。首次使用命名权重训练时，Ultralytics 可能会下载权重。任务完成后，`best.pt` 和 `last.pt` 会登记到 `data/models/` 下；模型工作区可以下载它们，或生成静态 FP32 ONNX 文件。

使用左侧边栏的“日志”页面查看本地后端运行日志的有限末尾内容。语言设置会改变工作区外壳以及新的设置/日志页面；该偏好只保存在浏览器中。

## 配置与数据位置

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `YWA_DATA_DIR` | `./data` | Starter 自有的 SQLite 数据、图片、运行目录、导出文件和模型的存储位置 |
| `YWA_IMPORT_ROOT` | `./data/imports` | 服务端目录扫描唯一允许使用的根目录 |
| `YWA_HOST` | `127.0.0.1` | 本地绑定地址；未配置身份认证时应保持本地绑定 |
| `YWA_MAX_UPLOAD_MB` | `50` | 单次上传大小限制 |
| `YWA_YOLO_EXECUTABLE` | 仓库 `.venv/bin/yolo` | 经过审核的 Ultralytics 可执行文件本地覆盖路径 |
| `YWA_SAM_MODEL` | 未设置 | 本地或 Ultralytics 可识别的 SAM 检查点；配置后启用真实的框提示和点提示推理 |
| `YWA_SAM_DEVICE` | `auto` | SAM 推理设备请求；如果 Ultralytics 提供解析结果，响应会报告实际设备 |
| `YWA_SAM_IMGSZ` | `1024` | SAM 推理图像尺寸 |

SAM 设置页面会将配置持久化到 `YWA_DATA_DIR/settings.json`；运行日志存储在 `YWA_DATA_DIR/logs/backend.log`，每个日志文件达到 2 MiB 后轮换并保留三个备份。日志页面展示最新内容时会合并这些日志文件。

不要将数据库、导入目录或模型注册表指向 Enterprise 仓库的工作副本。

## 故障排查

| 现象 | 处理方法 |
|---|---|
| 训练或校验出现 NumPy 兼容性错误 | Starter 固定使用 `ultralytics==8.4.115` 与 NumPy 1.x 基线。请在仓库 `.venv` 中重新安装项目依赖：`.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt`。 |
| 依赖解析器报告 NumPy 冲突 | 使用 Python 3.11 或 3.12 重新创建 `.venv`，然后安装固定版本依赖。Python 3.13 不在发布支持范围内。 |
| API 无法启动 | 确认 `YWA_HOST` 为 `127.0.0.1`，然后执行 `PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head`。 |
| 训练任务被拒绝 | 先校验数据集，确保 `train` 和 `val` 都包含已标注图片，并使用与数据集类型匹配的模型系列。 |
| 首次训练无法下载权重 | 为 Ultralytics 首次下载命名权重提供网络访问，或使用 `data/models/` 下已有的受管模型。 |
| ONNX 导出失败 | 将源 PT 文件保存在 `data/models/` 下，确认已安装 `onnx`、`onnxruntime`、`onnxscript` 和 `onnx_ir`，然后检查任务/模型响应中的错误信息。 |

## 已知限制

- 仅支持本地单用户使用；不提供身份认证、RBAC、TLS、公开部署或多租户能力。
- 仅支持 detect、segment、OBB 和单标签 classify；不支持 pose。
- 自动标注从数据集卡片进入：选择匹配任务类型的受管 PT，确认类别映射（未映射类别会跳过，不会按索引猜测），调整置信度/IoU 后启动；任务支持进度、日志和取消，生成结果会标记为自动来源，必须人工审核。训练与自动标注会在本机互斥执行。仍不提供 Agent、Workflow、evaluation、deployment 或文本提示分割。
- CPU 已通过发布测试。CUDA 单卡/多卡和 Apple MPS 属于尽力支持，必须在目标环境中进行冒烟测试；不支持远程训练调度器。
