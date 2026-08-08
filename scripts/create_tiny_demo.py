from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def write_example(root: Path) -> None:
    for split, color, count in (("train", "#3b82f6", 2), ("val", "#22c55e", 1)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.new("RGB", (128, 96), "#111827")
            draw = ImageDraw.Draw(image)
            offset = index * 8
            draw.rectangle((32 + offset, 24, 96 + offset, 72), fill=color)
            stem = f"{split}-{index + 1}"
            image.save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: square\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Tiny detect demo\n\nA generated, two-image YOLO detect dataset for local smoke tests.\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: create_tiny_demo.py OUTPUT_DIRECTORY")
    target = Path(sys.argv[1]).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {target}")
    write_example(target)
    print(f"Wrote tiny YOLO detect demo to {target}")
