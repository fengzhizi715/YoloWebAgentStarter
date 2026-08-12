from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.models import ImageItem
from app.core.storage import Storage


class DuplicateDetector:
    """Adapted from upstream DuplicateDetector for Starter managed image storage."""

    def analyze(self, images: list[ImageItem], storage: Storage, threshold: int = 8) -> dict:
        exact: dict[str, list[str]] = defaultdict(list)
        hashes: dict[str, int] = {}
        invalid: list[str] = []
        for image in images:
            try:
                path = storage.image_path(image.dataset_id, image.storage_name)
                exact[self._md5(path)].append(image.id)
                hashes[image.id] = self._average_hash(path)
            except (OSError, UnidentifiedImageError, ValueError):
                invalid.append(image.id)
        groups = [items for items in exact.values() if len(items) > 1]
        exact_matches = [{"canonical_image_id": group[0], "image_ids": group, "kind": "exact", "score": 1.0} for group in groups]
        excluded = {image_id for group in groups for image_id in group[1:]}
        representatives: dict[str, int] = {}
        similar: list[dict] = []
        for image_id, image_hash in hashes.items():
            if image_id in excluded:
                continue
            match = next((candidate for candidate, candidate_hash in representatives.items() if self._hamming(image_hash, candidate_hash) <= threshold), None)
            if match is None:
                representatives[image_id] = image_hash
                continue
            distance = self._hamming(image_hash, representatives[match])
            similar.append({"canonical_image_id": match, "image_ids": [match, image_id], "kind": "similar", "score": round(1 - distance / 64, 6), "hamming_distance": distance})
        return {"images": len(images), "duplicate": sum(len(group["image_ids"]) - 1 for group in exact_matches), "similar": len(similar), "invalid_images": len(invalid), "invalid_image_ids": invalid[:100], "phash_distance": threshold, "groups": exact_matches + similar}

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _average_hash(path: Path) -> int:
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((8, 8)).getdata())
        average = sum(pixels) / len(pixels)
        return sum(1 << index for index, pixel in enumerate(pixels) if pixel >= average)

    @staticmethod
    def _hamming(left: int, right: int) -> int:
        return (left ^ right).bit_count()
