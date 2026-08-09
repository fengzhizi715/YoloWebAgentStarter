from __future__ import annotations

import json
from collections import Counter
from importlib.metadata import Distribution, distributions
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "dependency-audit.md"
LOCKFILE_PATH = REPOSITORY_ROOT / "frontend" / "package-lock.json"


def python_license(distribution: Distribution) -> str:
    metadata = distribution.metadata
    classifiers = metadata.get_all("Classifier") or []
    license_classifiers = [item.removeprefix("License :: ") for item in classifiers if item.startswith("License :: ")]
    if license_classifiers:
        return normalize_license("; ".join(license_classifiers))
    declared = (metadata.get("License") or "").strip()
    if not declared or declared.upper() == "UNKNOWN":
        return "UNKNOWN"
    return declared_license_label(declared)


def declared_license_label(value: str) -> str:
    normalized = normalize_license(value)
    lowered = normalized.lower()
    if "mit license" in lowered or "permission is hereby granted" in lowered:
        return "MIT (declared text)"
    if "apache" in lowered:
        return "Apache-2.0 (declared text)"
    if "bsd" in lowered:
        return "BSD (declared text)"
    if "mozilla public license" in lowered:
        return "MPL-2.0 (declared text)"
    return normalized


def normalize_license(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) > 120:
        return f"{normalized[:117]}..."
    return normalized


def python_packages() -> list[tuple[str, str, str]]:
    packages = {
        (distribution.metadata["Name"].lower(), distribution.version, python_license(distribution))
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(packages)


def npm_packages() -> list[tuple[str, str, str]]:
    data = json.loads(LOCKFILE_PATH.read_text(encoding="utf-8"))
    packages = []
    for package_path, metadata in data["packages"].items():
        if not package_path:
            continue
        name = package_path.removeprefix("node_modules/")
        packages.append((name, metadata["version"], normalize_license(metadata.get("license", "UNKNOWN")) or "UNKNOWN"))
    return sorted(packages)


def section(title: str, packages: list[tuple[str, str, str]]) -> list[str]:
    licenses = Counter(license for _, _, license in packages)
    lines = [f"## {title}", "", f"Package count: **{len(packages)}**.", "", "License metadata summary:"]
    lines.extend(f"- `{license}`: {count}" for license, count in sorted(licenses.items()))
    lines.extend(["", "| Package | Version | License metadata |", "|---|---|---|"])
    lines.extend(f"| `{name}` | `{version}` | {license.replace('|', '\\|')} |" for name, version, license in packages)
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# Dependency audit",
        "",
        "Generated from the repository `.venv` package metadata and `frontend/package-lock.json`.",
        "License metadata is an inventory aid, not a legal determination; every `UNKNOWN` entry requires manual review before release.",
        "",
        *section("Python distributions", python_packages()),
        *section("npm packages", npm_packages()),
    ]
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
