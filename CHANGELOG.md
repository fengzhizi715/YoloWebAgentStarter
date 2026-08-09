# 更新日志

## 未发布

- 建立独立的 Starter 仓库与 SQLite baseline。
- 增加数据集、类别、图片、bbox/polygon 标注、校验与 YOLO 数据交换基础能力。
- 增加浏览器工作台：图片上传、受限目录扫描、手工标注、校验与 YOLO 导入/导出控制。
- 增加本地 detect/segment YOLO 训练任务：FIFO 队列、进程停止控制、日志、进度、摘要与 best/last checkpoint 下载。
- 增加从训练产物创建的受管模型版本、PT 下载、模型元数据生命周期与去重的 ONNX FP32 导出。
- Community v2 增加与固定 YoloWebAgent 快照兼容的 OBB 与 classify 数据集、标注、YOLO 交换和本地训练闭环。
- 增加 segment 的 SAM 辅助建议：可配置 Ultralytics SAM 本地/命名检查点；未配置时明确标识为 review-only box_stub。
- 修复导入根目录内文件软链接指向根目录外仍会被扫描读取的问题。
