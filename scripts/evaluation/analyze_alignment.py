"""Measure translational misalignment in the non-test PCB image pairs.

This is a diagnostic only.  It does not modify images, register them, or use
the held-out test split.  Phase correlation estimates the translation required
to place a defective image over its reference image; a near-zero translation
supports the raw-aligned baseline assumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image, preprocess_pair


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["split"] in {"development", "validation"}]


def estimate_translation(
    reference: np.ndarray,
    defective: np.ndarray,
    max_dimension: int,
) -> tuple[float, float, float]:
    """Return x shift, y shift and phase-correlation response without warping."""
    reference_gray, defective_gray = preprocess_pair(reference, defective)
    height, width = reference_gray.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        resized_size = (round(width * scale), round(height * scale))
        reference_gray = cv2.resize(reference_gray, resized_size, interpolation=cv2.INTER_AREA)
        defective_gray = cv2.resize(defective_gray, resized_size, interpolation=cv2.INTER_AREA)

    window = cv2.createHanningWindow(
        (reference_gray.shape[1], reference_gray.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(
        np.float32(reference_gray), np.float32(defective_gray), window
    )
    return float(shift[0] / scale), float(shift[1] / scale), float(response)


def summary(records: list[dict[str, object]]) -> dict[str, object]:
    magnitudes = [float(record["shift_magnitude_px"]) for record in records]
    responses = [float(record["phase_response"]) for record in records]
    return {
        "images": len(records),
        "splits": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "median_shift_magnitude_px": statistics.median(magnitudes),
        "mean_shift_magnitude_px": statistics.fmean(magnitudes),
        "p95_shift_magnitude_px": float(np.percentile(magnitudes, 95)),
        "max_shift_magnitude_px": max(magnitudes),
        "images_with_shift_over_1px": sum(value > 1.0 for value in magnitudes),
        "images_with_shift_over_2px": sum(value > 2.0 for value in magnitudes),
        "median_phase_response": statistics.median(responses),
        "mean_phase_response": statistics.fmean(responses),
    }


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "dataset_split.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "alignment_diagnostic.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "alignment_diagnostic_summary.json",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=512,
        help="Maximum diagnostic image dimension; translations are reported in original pixels.",
    )
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if not rows:
        raise ValueError("No development/validation records found in manifest.")
    if args.max_dimension <= 0:
        raise ValueError("--max-dimension must be positive.")

    records: list[dict[str, object]] = []
    for row in rows:
        reference = load_image(resolve_project_path(row["reference_path"]))
        defective = load_image(resolve_project_path(row["image_path"]))
        shift_x, shift_y, response = estimate_translation(
            reference, defective, args.max_dimension
        )
        records.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "defect_class": row["defect_class"],
                "shift_x_px": shift_x,
                "shift_y_px": shift_y,
                "shift_magnitude_px": float(np.hypot(shift_x, shift_y)),
                "phase_response": response,
            }
        )

    records.sort(key=lambda record: str(record["image_id"]))
    write_csv(records, args.output)
    report = summary(records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Per-image diagnostic: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
