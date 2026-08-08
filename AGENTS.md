# YoloWebAgentStarter contributor notes

YoloWebAgentStarter is the independent community edition of YoloWebAgent. It must not import, read, or depend on the Enterprise repository at runtime.

## Product boundary

- Starter v1 supports `detect` and `segment` only.
- Core flow: images → dataset → bbox/polygon annotation → validation → YOLO import/export → local training → managed PT/ONNX artifacts.
- Training is local and queue-backed; model versions are created only from managed training artifacts.
- Auth, RBAC, License, Agent, Workflow, Evaluation, Deployment, SAM, automatic annotation, OBB, pose, and classify are out of scope.
- The service is local single-user software and binds to `127.0.0.1` by default.

## Engineering rules

- Python dependencies belong in the repository `.venv`; never install into system Python.
- API routes only adapt HTTP requests. Business logic belongs in domain services.
- File IO goes through `backend/app/core/storage.py` or a named domain boundary.
- Persist annotation coordinates in absolute image pixels. Canvas state is not a persistence schema.
- Reuse persisted image splits in validation and YOLO export.
- New schema changes require Alembic migrations. Do not add runtime SQLite patchers.
- Frontend API calls belong under `frontend/src/api/`; page components do not hardcode API URLs.
- Keep source attribution in `source_snapshot.md`; never copy the Enterprise `.git` directory or migration history.

## Verification

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```
