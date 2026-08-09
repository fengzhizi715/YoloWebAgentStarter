# Five-minute quick start

## 1. Start locally

Use Python 3.11 or 3.12 and Node.js 20 or newer. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
npm --prefix frontend install
./run-backend.sh
```

Start the frontend in another terminal:

```bash
./run-frontend.sh
```

Open `http://127.0.0.1:5173`. Both development servers are local-only by default. Alternatively, run `./run-all.sh` from the repository root to start both processes; `Ctrl+C` stops them together.

## 2. Create data and annotate

1. Create a detect or segment dataset.
2. Upload images, or put them below `data/imports/` and scan a relative subdirectory.
3. Add one or more classes and annotate each image with boxes or polygons.
4. Keep at least one image in both `train` and `val`, then run dataset validation.
5. Export/import YOLO ZIP files from the dataset workspace when needed.

For a disposable tiny detect dataset that can be imported through the UI:

```bash
./.venv/bin/python scripts/create_tiny_demo.py /tmp/ywa-tiny-demo
```

The command writes three generated PNGs, labels, and `data.yaml`; it does not download or redistribute a model weight.

## 3. Train and export

Open the training workspace, choose a matching named model (for example `yolo11n.pt` for detect), and start a local task. The first named-weight run may download a weight through Ultralytics. Finished tasks register `best.pt` and `last.pt` beneath `data/models/`; the model workspace can download them or create a static FP32 ONNX file.

## Configuration and data locations

| Setting | Default | Purpose |
|---|---|---|
| `YWA_DATA_DIR` | `./data` | All Starter-owned SQLite data, images, runs, exports, and models |
| `YWA_IMPORT_ROOT` | `./data/imports` | Only root permitted for server-side directory scanning |
| `YWA_HOST` | `127.0.0.1` | Local bind address; keep it local without adding authentication |
| `YWA_MAX_UPLOAD_MB` | `50` | Per-upload size limit |
| `YWA_YOLO_EXECUTABLE` | repository `.venv/bin/yolo` | Reviewed local override for the Ultralytics executable |

Never point the database, import directory, or model registry at an Enterprise checkout.

## Troubleshooting

| Symptom | Action |
|---|---|
| Dependency resolver reports NumPy conflicts | Recreate `.venv` with Python 3.11 or 3.12, then install the pinned requirements. Python 3.13 is not a release target. |
| API will not start | Check `YWA_HOST` is `127.0.0.1`, then run `PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head`. |
| Training is rejected | Validate the dataset, ensure both `train` and `val` contain images, and use a model family matching the dataset type. |
| First training run cannot download a weight | Supply network access for Ultralytics' first named-weight download, or use an already managed model under `data/models/`. |
| ONNX export fails | Keep the source PT under `data/models/`, confirm the installed `onnx`, `onnxruntime`, `onnxscript`, and `onnx_ir` packages, then inspect the task/model response error. |

## Known limits

- Local single-user only; no authentication, RBAC, TLS, public deployment, or tenancy.
- Only detect and segment are supported.
- No automatic annotation, Agent, Workflow, evaluation, deployment, OBB, pose, or classify.
- CPU is release-tested. GPU and Apple MPS are best-effort and must be smoke-tested in the target environment.
