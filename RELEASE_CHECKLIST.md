# Release checklist

The following checks are required before a public tag.

- [ ] Install from a new Python 3.11/3.12 virtual environment and an empty data directory.
- [ ] Run `PYTHONPATH=backend .venv/bin/pytest backend/tests` and `npm --prefix frontend test`.
- [ ] Run `npm --prefix frontend run build`.
- [ ] Complete a real CPU detect training and ONNX export smoke using a temporary data directory.
- [ ] Re-run the optional GPU/MPS smoke on every platform claimed by the release.
- [ ] Scan tracked files for secrets, private addresses, build logs, and unintended absolute paths.
- [ ] Review Python/npm licenses and Ultralytics/model-weight distribution terms.
- [ ] Verify upload, import scan, image serving, task artifact, model download, and ONNX export cannot leave managed roots.
- [ ] Confirm the application binds to `127.0.0.1` and the README still warns against public deployment.
- [ ] Add an approved Enterprise contact URL to the README; this repository currently has no verified one.
