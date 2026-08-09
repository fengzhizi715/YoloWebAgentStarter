# YoloWebAgentStarter

YoloWebAgentStarter 是用于构建 YOLO 数据集的本地开源工作台。首个公开版本聚焦于可靠的数据闭环：

```text
图片 → 数据集 → bbox / polygon 标注 → 校验 → YOLO 导入 / 导出
```

Starter 仓库独立于 YoloWebAgent Enterprise，拥有自己的 Git 历史、SQLite 数据库、受管数据目录、API 与前端构建产物。

## 当前范围

- 数据集、类别、图片与 split 管理
- 浏览器图片上传与受限的本地目录扫描
- detect 的 bbox 标注与 segment 的 polygon 标注
- 数据集校验
- YOLO detect/segment ZIP 导入与导出
- 本地 YOLO detect/segment 训练：队列任务、日志、进度、停止控制与 best/last checkpoint
- 从训练产物创建的受管模型版本、PT 下载与 ONNX FP32 导出
- 本地单用户运行，默认绑定 `127.0.0.1`

Agent 工作流、自动标注、评估、部署运行时、OBB、pose 与 classify 不属于当前实现范围。

## 快速开始

要求：Python 3.11 或 3.12（已使用 3.12 完成发布验证），Node.js 20+。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
npm --prefix frontend install
./scripts/run-backend.sh
```

在另一个终端中运行：

```bash
./scripts/run-frontend.sh
```

打开 <http://127.0.0.1:5173>，API 监听 <http://127.0.0.1:8000>。

请使用 Python 3.11 或 3.12 创建 `.venv`。首次使用 `yolo11n.pt` 等命名 YOLO 权重时，Ultralytics 可能下载该权重。请勿将模型权重放在任意文件系统位置：训练任务仅接受 `data/models/` 受管目录中的本地路径。

请参阅 [docs/quick-start.md](docs/quick-start.md) 获取五分钟上手、配置示例与故障排查。执行 `./.venv/bin/python scripts/create_tiny_demo.py /tmp/ywa-tiny-demo` 可生成一个临时的微型 YOLO 示例数据集。

后端启动时会将 SQLite 数据库升级到当前 Alembic revision。如需显式执行：

```bash
PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head
```

训练会使用数据集中持久化的 `train` 与 `val` 图片 split。请确保每个 split 至少包含一张图片，完成数据集校验后，再打开 `训练` 工作区。Ultralytics 随后端依赖一起安装；首次使用 `yolo11n.pt` 等命名权重时可能下载该权重。

训练完成后，`best.pt` 与 `last.pt` 会自动登记到受管模型目录。打开 `模型` 工作区可编辑元数据、归档或恢复模型、下载 PT，或创建去重的 ONNX FP32 导出。

## 数据与导入边界

默认情况下，所有运行时状态均写入 `./data`，且该目录已被 Git 忽略。浏览器上传的图片会复制到受管数据集目录。

服务端目录扫描仅允许访问 `YWA_IMPORT_ROOT`（默认：`./data/imports`）以内的路径。请将图片放在该目录下，并在 UI 中提交相对路径；导入根目录外的绝对路径会被拒绝。

配置项见 [backend/.env.example](backend/.env.example)。

## 仅限本地的安全边界

Starter 没有认证、授权、TLS 终止或多用户隔离能力。它刻意默认绑定到 `127.0.0.1`，**不是**可直接公网部署的模板。不要绑定到 `0.0.0.0`、转发端口，或在未补充适当安全层的情况下放到公网反向代理之后。发布构建前请阅读 [SECURITY.md](SECURITY.md) 与 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。

## Starter 与 Enterprise

| 能力 | Starter | Enterprise |
|---|---|---|
| detect 与 segment 数据集流程 | 本地单用户 | 协作式且受治理的工作流 |
| 训练与模型产物 | 本地 FIFO 队列；PT 与静态 FP32 ONNX | 受管自动化、评估、部署与运维 |
| 访问控制与部署 | 不提供 | 提供 Enterprise 专属控制与交付能力 |
| 自动标注、Agent、Workflow、评估 | 不提供 | Enterprise 产品线提供 |

### Enterprise 联系入口

> Enterprise 官网正在备案中，正式的产品介绍与联系入口将在备案完成后公布。

在正式 URL 发布前，常规 issue 渠道仅用于 Starter 的缺陷反馈；维护者仍需在营销发布前替换为已批准的销售或联系 URL。

## 开发检查

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

产品边界与迁移来源请参见 [phase1_scope.md](phase1_scope.md)、[migration_matrix.md](migration_matrix.md) 与 [source_snapshot.md](source_snapshot.md)。第三方依赖与模型权重的分发说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已知限制

- 这是本地单用户应用，没有认证能力，不应直接暴露到公网。
- 当前版本不包含自动标注、评估、部署、OBB、pose 或 classify。
- 目录扫描仅接受 `YWA_IMPORT_ROOT` 以下路径；浏览器上传是可移植的默认方式。
