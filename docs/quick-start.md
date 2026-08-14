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

Open `http://127.0.0.1:5173`. Both development servers are local-only by default. Alternatively, run `./run-all.sh` (or `sh run-all.sh`) from the repository root to start both processes; `Ctrl+C` stops them together.

## 2. Create data and annotate

1. Create a `detect`, `segment`, `obb`, or `classify` dataset.
2. Upload images, or put them below `data/imports/` and scan a relative subdirectory.
3. Add one or more classes, then use bbox (detect), polygon (segment), rotated box (OBB), or one image-level class (classify).
4. Keep at least one image in both `train` and `val`, then run dataset validation.
5. Export/import YOLO ZIP files for detect, segment, or OBB; classification uses the standard `train/<class>/<image>` directory layout.

For segment datasets, open the left sidebar's SAM settings page to configure the model, inference device, image size, and fallback mode. The same values can be supplied as `YWA_SAM_*` environment defaults. Without a model, the UI disables point prompts and clearly labels box prompts as a review-only rectangular suggestion; it does not claim model inference or save the suggestion as SAM-generated.

For OBB datasets, drag on an empty canvas area to create a rotated box. Click an existing OBB to select it, drag it to move it, use the four corner handles to resize it, and use the top rotation handle (or the angle field) to rotate it. The editor keeps the resulting rotated corners inside the image before saving.

For a disposable tiny detect dataset that can be imported through the UI:

```bash
./.venv/bin/python scripts/create_tiny_demo.py /tmp/ywa-tiny-demo
```

The command writes three generated PNGs, labels, and `data.yaml`; it does not download or redistribute a model weight.

## 3. Train and export

Open the training workspace, choose a matching named model (for example `yolo11n.pt`, `yolo11n-seg.pt`, `yolo11n-obb.pt`, or `yolo11n-cls.pt`), select CPU, MPS, or CUDA. When multiple CUDA devices are available, select two or more GPUs to pass `device=0,1` to local Ultralytics DDP. The first named-weight run may download a weight through Ultralytics. Finished tasks register `best.pt` and `last.pt` beneath `data/models/`; the model workspace can download them or create a static FP32 ONNX file.

Use the left sidebar's Logs page to inspect the bounded tail of the local backend runtime log. The language setting changes the workspace shell and the new settings/logs pages; the preference is stored only in the browser.

## Configuration and data locations

| Setting | Default | Purpose |
|---|---|---|
| `YWA_DATA_DIR` | `./data` | All Starter-owned SQLite data, images, runs, exports, and models |
| `YWA_IMPORT_ROOT` | `./data/imports` | Only root permitted for server-side directory scanning |
| `YWA_HOST` | `127.0.0.1` | Local bind address; keep it local without adding authentication |
| `YWA_MAX_UPLOAD_MB` | `50` | Per-upload size limit |
| `YWA_YOLO_EXECUTABLE` | repository `.venv/bin/yolo` | Reviewed local override for the Ultralytics executable |
| `YWA_SAM_MODEL` | unset | Local or Ultralytics-recognized SAM checkpoint; enables actual box and point inference |
| `YWA_SAM_DEVICE` | `auto` | SAM inference device request; responses report the resolved device when Ultralytics exposes it |
| `YWA_SAM_IMGSZ` | `1024` | SAM inference image size |

The SAM settings page persists to `YWA_DATA_DIR/settings.json`; runtime logs are stored in `YWA_DATA_DIR/logs/backend.log`.

Never point the database, import directory, or model registry at an Enterprise checkout.

## Troubleshooting

| Symptom | Action |
|---|---|
| Training fails with `AttributeError: module 'numpy' has no attribute 'trapz'` | The environment has NumPy 2.x, which is incompatible with the pinned `ultralytics==8.3.40` validation code. Reinstall the project requirements in the repository `.venv` to use NumPy 1.x: `.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt`. |
| Dependency resolver reports NumPy conflicts | Recreate `.venv` with Python 3.11 or 3.12, then install the pinned requirements. Python 3.13 is not a release target. |
| API will not start | Check `YWA_HOST` is `127.0.0.1`, then run `PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head`. |
| Training is rejected | Validate the dataset, ensure both `train` and `val` contain images, and use a model family matching the dataset type. |
| First training run cannot download a weight | Supply network access for Ultralytics' first named-weight download, or use an already managed model under `data/models/`. |
| ONNX export fails | Keep the source PT under `data/models/`, confirm the installed `onnx`, `onnxruntime`, `onnxscript`, and `onnx_ir` packages, then inspect the task/model response error. |

## Known limits

- Local single-user only; no authentication, RBAC, TLS, public deployment, or tenancy.
- Supports detect, segment, OBB, and single-label classify only; pose is not supported.
- No batch automatic annotation, Agent, Workflow, evaluation, deployment, or text-prompt segmentation.
- CPU is release-tested. CUDA single/multi-GPU and Apple MPS are best-effort and must be smoke-tested in the target environment; remote training schedulers are not supported.
