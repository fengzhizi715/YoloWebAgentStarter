# Release checklist

The following checks are required before a public tag.

- [ ] Obtain and record a rights-holder authorization or an applicable upstream license for commit `701f6e5a63b73f39e35f48fb6de7d2414401875a` in `docs/provenance/UPSTREAM_AUTHORIZATION.md`. The tag gate must pass.
- [ ] Install from a new Python 3.11/3.12 virtual environment and an empty data directory.
- [ ] Run `PYTHONPATH=backend .venv/bin/pytest backend/tests` and `npm --prefix frontend test`.
- [ ] Run `npm --prefix frontend run build`.
- [ ] Complete `PYTHONPATH=backend .venv/bin/python scripts/run_cpu_smoke.py`, which performs real CPU detect, segment, OBB, classify training and detect ONNX export using temporary data.
- [ ] Re-run the optional GPU/MPS smoke on every platform claimed by the release.
- [ ] Scan tracked files for secrets, private addresses, build logs, and unintended absolute paths.
- [ ] Review Python/npm licenses and Ultralytics/model-weight distribution terms.
- [ ] Verify upload, import scan, image serving, task artifact, model download, and ONNX export cannot leave managed roots.
- [ ] Verify the YOLO ZIP limits reject excess member count, member size, total uncompressed size, and compression ratio.
- [ ] Confirm the application binds to `127.0.0.1` and the README still warns against public deployment.
- [ ] Enable GitHub private vulnerability reporting or add a maintainer-controlled security email, then record and test the private channel in `SECURITY.md`.
