# Community v2 migration matrix

This matrix is a file-level provenance aid, not a license grant. The exact upstream baseline is commit `701f6e5a63b73f39e35f48fb6de7d2414401875a`; public release remains blocked until the rights evidence in [`docs/provenance/UPSTREAM_AUTHORIZATION.md`](docs/provenance/UPSTREAM_AUTHORIZATION.md) is approved.

| Upstream reference area | Community v2 implementation | Relationship and retained scope | Excluded upstream concerns |
|---|---|---|---|
| `backend/app/core/task_types.py`, `core/schemas.py`, `dataset/annotation/service.py` | `backend/app/core/task_types.py`, `backend/app/core/schemas.py`, `backend/app/dataset/annotation/` | detect, segment/polygon, OBB and classify task/annotation contracts; Community persists absolute-pixel coordinates in its own SQLite schema. | Auth, RBAC, License, profiles and Enterprise tables. |
| `dataset/exchange/yolo/dataset_file_writer.py`, `label_writer.py` | `backend/app/dataset/exchange/yolo.py` | YOLO detect/segment/OBB labels and classify directory layout; Community adds archive traversal, ZIP resource and managed-storage boundaries. | Batch preparation, provider integrations and Enterprise exports. |
| `sam/schemas.py`, `sam/backends/box_stub.py`, `sam/ultralytics_backend.py` | `backend/app/sam/` and `frontend/src/annotation/AnnotationCanvas.tsx` | Box/point prompt and reviewable polygon-suggestion contract; SAM output remains a user-confirmed normal annotation. | Text prompts, batch auto-labeling, provider/model profiles. |
| Annotation UI concepts | `frontend/src/annotation/` and `frontend/src/App.tsx` | Task-specific canvas and OBB selection/rotation/resize interaction are kept local to this app and API. | Enterprise routing, collaboration, workflow and deployment UI. |
| Training task concepts | `backend/app/training/`, `scripts/run_cpu_smoke.py` | Local queue, persisted split export and managed PT/ONNX artifacts; smoke covers detect, segment, OBB and classify. | Remote workers, evaluation, deployment and agent execution. |

All other Community files must be treated as Starter-owned only after they are covered by the same authorization decision or a separately documented clean-room audit. No Starter runtime imports, reads, or depends on the upstream repository.
