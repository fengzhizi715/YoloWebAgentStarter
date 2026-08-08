# Third-party and model distribution notes

This file is a release checklist, not legal advice. Before a public release, the publisher must review the exact package and model versions that will be shipped.

## What this repository distributes

- Source code is MIT-licensed; see `LICENSE` and `NOTICE`.
- Python dependencies are installed by `pip` from `backend/requirements*.txt`; wheels are not committed or redistributed by this repository.
- Frontend dependencies are resolved from `frontend/package-lock.json`; `node_modules` is not committed or redistributed.
- No pretrained weights, customer datasets, screenshots, videos, vendor wheels, or model artifacts are committed. Named Ultralytics weights are downloaded only on first use.
- Generated demo images come from `scripts/create_tiny_demo.py` and are entirely local.

## Release-sensitive dependency

`ultralytics==8.3.40` enables training and ONNX export. Ultralytics documents AGPL-3.0 and an alternative Enterprise License for its software and AI models. A publisher must confirm that the intended distribution and any downloaded or trained weights comply with the applicable terms before shipping this Starter or a derivative.

## Required release review

1. Generate and retain an SBOM or package-license report for the exact Python and npm lockfiles used in the release.
2. Do not add wheels, weights, exported models, datasets, screenshots, or videos until their origin and redistribution terms are recorded.
3. Re-run a secret scan and confirm that no private endpoints, customer records, local absolute paths, or commercial-license files are added.
4. Keep the upstream provenance described in `source_snapshot.md` and `NOTICE`.
