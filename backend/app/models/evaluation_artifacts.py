from __future__ import annotations

from pathlib import Path


class EvaluationArtifactManager:
    """Community port of upstream app/evaluation/artifacts.py."""

    def find_artifacts(self, run_dir: str | Path) -> dict[str, str | None]:
        run_path = Path(run_dir)
        confusion = self._first_existing(run_path, ["confusion_matrix.png", "confusion_matrix_normalized.png"])
        generic_pr_curve = self._first_existing(run_path, ["PR_curve.png", "P_curve.png", "R_curve.png"])
        box_pr_curve = self._first_existing(run_path, ["BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"])
        mask_pr_curve = self._first_existing(run_path, ["MaskPR_curve.png", "MaskP_curve.png", "MaskR_curve.png"])
        # Preserve the upstream generic key for detect/OBB and existing clients;
        # segment in Ultralytics 8.3.40 prefixes the same artifact with Box/Mask.
        pr_curve = generic_pr_curve or box_pr_curve or mask_pr_curve
        predictions = run_path / "predictions.json"
        return {
            "confusion_matrix": str(confusion) if confusion else None,
            "pr_curve": str(pr_curve) if pr_curve else None,
            "box_pr_curve": str(box_pr_curve) if box_pr_curve else None,
            "mask_pr_curve": str(mask_pr_curve) if mask_pr_curve else None,
            "predictions": str(predictions) if predictions.exists() else None,
        }

    @staticmethod
    def _first_existing(root: Path, names: list[str]) -> Path | None:
        for name in names:
            path = root / name
            if path.exists():
                return path
        return None
