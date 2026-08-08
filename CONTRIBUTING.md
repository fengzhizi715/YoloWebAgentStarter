# Contributing

Please keep changes inside the Starter scope documented in `phase1_scope.md`. Open an issue before adding a new task type, authentication, cloud services, or Enterprise-only workflow.

Use a repository-local Python virtual environment, add focused tests, and run:

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

Do not include datasets, model weights, credentials, customer material, or absolute local paths in commits.

