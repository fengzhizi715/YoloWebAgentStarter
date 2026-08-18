# 上游授权发布门槛

```text
Release status: pending
Upstream commit: 701f6e5a63b73f39e35f48fb6de7d2414401875a
Rights holder:
Approved date:
Evidence reference:
```

公开发布目前被阻断。

记录的 YoloWebAgent 基线来自 `https://github.com/fengzhizi715/YoloWebAgent.git` 的 commit `701f6e5a63b73f39e35f48fb6de7d2414401875a`。在记录该快照时，其顶层目录没有 `LICENSE` 或 `NOTICE`。本地 MIT 文件不能为该代码树创建再许可权利。

在创建公开 tag 之前，版权所有者或授权代表必须用可验证的证据替换此模板，且证据必须包含以下全部内容：

1. `Status: approved`。
2. 上述完整 commit SHA（或经过单独批准的替代 SHA）。
3. 适用的上游许可证文本，或明确允许选择性派生及其目标公开许可证的书面授权。
4. 权利人/授权代表、日期和持久化引用（签署文件、公开许可证 URL 或归档通信标识符）。
5. 确认 [`../../migration_matrix.md`](../../migration_matrix.md) 中的文件级范围已被覆盖，或列出排除/重写的文件。

不要将私人通信、签名、客户数据或凭据放入公共仓库。底层证据应存放在维护者控制的记录系统中，这里只保留不含敏感信息的持久化引用。在本文档按要求标记为 approved 并填写必需字段之前，`scripts/check_release_provenance.py` 会阻止 tag 的 CI 流程通过。
