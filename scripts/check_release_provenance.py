"""Fail release automation until upstream derivative rights are recorded."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = ROOT / "docs" / "provenance" / "UPSTREAM_AUTHORIZATION.md"
EXPECTED_FIELDS = {
    "Release status": "approved",
    "Upstream commit": "701f6e5a63b73f39e35f48fb6de7d2414401875a",
}
REQUIRED_NONEMPTY_FIELDS = ("Rights holder", "Approved date", "Evidence reference")


def _fields(content: str) -> dict[str, str]:
    return {
        name.strip(): value.strip()
        for line in content.splitlines()
        if ":" in line
        for name, value in [line.split(":", maxsplit=1)]
    }


def main() -> int:
    content = AUTHORIZATION.read_text(encoding="utf-8") if AUTHORIZATION.is_file() else ""
    fields = _fields(content)
    missing = [f"{name}: {value}" for name, value in EXPECTED_FIELDS.items() if fields.get(name) != value]
    missing.extend(f"{name}:" for name in REQUIRED_NONEMPTY_FIELDS if not fields.get(name))
    if not missing:
        return 0
    print("Public release blocked: upstream authorization evidence is incomplete.", file=sys.stderr)
    print(f"Update {AUTHORIZATION.relative_to(ROOT)} with: {', '.join(missing)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
