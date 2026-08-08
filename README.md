# YoloWebAgentStarter

YoloWebAgentStarter is a local, open-source workspace for building YOLO datasets. The first public slice focuses on a dependable data loop:

```text
images → dataset → bbox / polygon annotation → validation → YOLO import / export
```

The Starter repository is independent from YoloWebAgent Enterprise. It has its own Git history, SQLite database, managed data directory, API, and frontend build.

## Current scope

- Dataset, class, image, and split management
- Browser image upload and restricted local-directory scanning
- Detect bbox and segment polygon annotation
- Dataset validation
- YOLO detect/segment ZIP import and export
- Local YOLO detect/segment training with queued tasks, logs, progress, stop control, and best/last checkpoints
- Managed model versions from training artifacts, PT downloads, and ONNX FP32 export
- Local single-user operation, bound to `127.0.0.1` by default

Agent workflows, automatic annotation, evaluation, deployment runtimes, OBB, pose, and classify are not part of this implementation slice.

## Quick start

Requirements: Python 3.10+, Node.js 20+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
npm --prefix frontend install
./scripts/run-backend.sh
```

In another terminal:

```bash
./scripts/run-frontend.sh
```

Open <http://127.0.0.1:5173>. The API listens on <http://127.0.0.1:8000>.

The backend upgrades the SQLite database to the current Alembic revision at startup. To run it explicitly:

```bash
PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head
```

Training uses the dataset's persisted `train` and `val` image splits. Create at least one image in each split, validate the dataset, then open the `训练` workspace. Ultralytics is installed with the backend requirements; the first use of a named weight such as `yolo11n.pt` may download that weight.

Completed training automatically registers `best.pt` and `last.pt` under the managed model directory. Open the `模型` workspace to edit metadata, archive or restore a model, download PT, or create a deduplicated ONNX FP32 export.

## Data and import boundaries

By default, all runtime state is placed in `./data` and is ignored by Git. Browser uploads are copied into the managed dataset directory.

Server-side directory scanning is restricted to `YWA_IMPORT_ROOT` (default: `./data/imports`). Put images below that directory and submit a relative path from the UI. Absolute paths outside the import root are rejected.

Configuration is documented in [backend/.env.example](backend/.env.example).

## Development checks

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

See [phase1_scope.md](phase1_scope.md), [migration_matrix.md](migration_matrix.md), and [source_snapshot.md](source_snapshot.md) for the product boundary and migration provenance.

## Known limitations

- This is a local, single-user application. It has no authentication and should not be exposed directly to the public internet.
- The current slice does not include automatic annotation, evaluation, deployment, OBB, pose, or classify.
- The directory scanner accepts paths only below `YWA_IMPORT_ROOT`; browser uploads are the portable default.
