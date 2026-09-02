"""Run raw ORB feature matching on the manifest's development split only."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.orb import detect_orb

ALGORITHM_VERSION = "raw-orb-v1"
DEVELOPMENT_COUNT = 139
DEVELOPMENT_BASELINE = {
    "n_features": 5000,
    "scale_factor": 1.2,
    "n_levels": 8,
    "spatial_distance_threshold": 15.0,
    "hamming_threshold": 60.0,
    "box_radius": 35,
    "matcher_type": "bf_crosscheck",
}
RESULT_FIELDS = [
    "image_id",
    "defect_class",
    "split",
    "algorithm_version",
    "num_reference_keypoints",
    "num_defective_keypoints",
    "num_matches",
    "num_consistent_matches",
    "num_inconsistent_matches",
    "num_unmatched_keypoints",
    "predicted_count",
    "ground_truth_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1_score",
    "mean_matched_iou",
    "iou_threshold",
    "processing_time_ms",
    "predicted_boxes_path",
    "ground_truth_boxes",
    "status",
    "error",
]


def load_development_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "development"
        ]
    if len(rows) != DEVELOPMENT_COUNT:
        raise ValueError(
            f"Expected {DEVELOPMENT_COUNT} development records, found {len(rows)}"
        )
    return sorted(rows, key=lambda row: row["image_id"])


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=16)
def load_reference(path: Path) -> np.ndarray:
    """Cache the read-only reference images during a dataset run."""
    return load_image(path)


def evaluate_record(
    row: dict[str, str],
    iou_threshold: float,
    boxes_directory: Path | None = None,
    n_features: int = 5000,
    spatial_distance_threshold: float = 15.0,
    hamming_threshold: float = 60.0,
    box_radius: int = 35,
) -> dict[str, str | int | float]:
    result: dict[str, str | int | float] = {
        "image_id": row["image_id"],
        "defect_class": row["defect_class"],
        "split": row["split"],
        "algorithm_version": ALGORITHM_VERSION,
        "status": "error",
        "error": "",
    }
    try:
        reference = load_reference(resolve_project_path(row["reference_path"]))
        defective = load_image(resolve_project_path(row["image_path"]))
        ground_truth_boxes = parse_voc_boxes(
            resolve_project_path(row["annotation_path"])
        )
        detection = detect_orb(
            reference,
            defective,
            n_features=n_features,
            spatial_distance_threshold=spatial_distance_threshold,
            hamming_threshold=hamming_threshold,
            box_radius=box_radius,
        )
        metrics = evaluate_boxes(
            detection.boxes,
            ground_truth_boxes,
            iou_threshold=iou_threshold,
        )
        predicted_boxes_path = ""
        if boxes_directory is not None:
            if not boxes_directory.is_absolute():
                boxes_directory = PROJECT_ROOT / boxes_directory
            boxes_directory.mkdir(parents=True, exist_ok=True)
            boxes_path = boxes_directory / f"{row['image_id']}.npy"
            box_dtype = np.dtype(
                [
                    ("xmin", "<i4"),
                    ("ymin", "<i4"),
                    ("xmax", "<i4"),
                    ("ymax", "<i4"),
                ]
            )
            compact_boxes = np.fromiter(
                (
                    (
                        box["xmin"],
                        box["ymin"],
                        box["xmax"],
                        box["ymax"],
                    )
                    for box in detection.boxes
                ),
                dtype=box_dtype,
                count=len(detection.boxes),
            )
            np.save(boxes_path, compact_boxes, allow_pickle=False)
            predicted_boxes_path = boxes_path.relative_to(PROJECT_ROOT).as_posix()

        result.update(
            {
                "num_reference_keypoints": detection.num_reference_keypoints,
                "num_defective_keypoints": detection.num_defective_keypoints,
                "num_matches": detection.num_matches,
                "num_consistent_matches": detection.num_consistent_matches,
                "num_inconsistent_matches": detection.num_inconsistent_matches,
                "num_unmatched_keypoints": detection.num_unmatched_keypoints,
                "predicted_count": len(detection.boxes),
                "ground_truth_count": len(ground_truth_boxes),
                **metrics,
                "processing_time_ms": detection.processing_time_ms,
                "predicted_boxes_path": predicted_boxes_path,
                "ground_truth_boxes": json.dumps(
                    ground_truth_boxes, separators=(",", ":")
                ),
                "status": "success",
            }
        )
    except Exception as error:  # Preserve one result record for every manifest row.
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def evaluate_records(
    rows: list[dict[str, str]],
    iou_threshold: float,
    workers: int = 1,
    boxes_directory: Path | None = None,
    n_features: int = 5000,
    spatial_distance_threshold: float = 15.0,
    hamming_threshold: float = 60.0,
    box_radius: int = 35,
) -> list[dict[str, str | int | float]]:
    """Evaluate independent records concurrently while preserving manifest order."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        results = []
        for index, row in enumerate(rows, start=1):
            results.append(
                evaluate_record(
                    row,
                    iou_threshold,
                    boxes_directory,
                    n_features=n_features,
                    spatial_distance_threshold=spatial_distance_threshold,
                    hamming_threshold=hamming_threshold,
                    box_radius=box_radius,
                )
            )
            if index % 20 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda row: evaluate_record(
                    row,
                    iou_threshold,
                    boxes_directory,
                    n_features=n_features,
                    spatial_distance_threshold=spatial_distance_threshold,
                    hamming_threshold=hamming_threshold,
                    box_radius=box_radius,
                ),
                rows,
            )
        )


def summarise(
    results: list[dict[str, str | int | float]], iou_threshold: float
) -> dict[str, object]:
    successful = [result for result in results if result["status"] == "success"]
    runtimes = [float(result["processing_time_ms"]) for result in successful]

    def aggregate_metrics(rows: list[dict[str, str | int | float]]) -> dict[str, float | int]:
        true_positives = sum(int(row["true_positives"]) for row in rows)
        false_positives = sum(int(row["false_positives"]) for row in rows)
        false_negatives = sum(int(row["false_negatives"]) for row in rows)
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        return {
            "images": len(rows),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "mean_image_iou": (
                statistics.fmean(float(row["mean_matched_iou"]) for row in rows)
                if rows
                else 0.0
            ),
        }

    by_class: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for result in successful:
        by_class[str(result["defect_class"])].append(result)

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "split": "development",
        "development_baseline": DEVELOPMENT_BASELINE,
        "legacy_raw_validation_record": (
            "configs/legacy_raw_validation/orb_validation_record.json"
        ),
        "iou_threshold": iou_threshold,
        "predicted_boxes_format": "NumPy structured array: xmin,ymin,xmax,ymax int32",
        "records": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "class_distribution": dict(
            sorted(Counter(str(row["defect_class"]) for row in results).items())
        ),
        "overall_box_metrics": aggregate_metrics(successful),
        "box_metrics_by_class": {
            defect_class: aggregate_metrics(class_rows)
            for defect_class, class_rows in sorted(by_class.items())
        },
        "runtime_ms": {
            "mean": statistics.fmean(runtimes) if runtimes else 0.0,
            "standard_deviation": statistics.pstdev(runtimes) if runtimes else 0.0,
            "minimum": min(runtimes) if runtimes else 0.0,
            "maximum": max(runtimes) if runtimes else 0.0,
            "frames_per_second": (
                1000.0 / statistics.fmean(runtimes) if runtimes and statistics.fmean(runtimes) > 0 else 0.0
            ),
        },
    }


def write_results(
    results: list[dict[str, str | int | float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset_split.csv",
        help="Path to the shared manifest CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "orb_development.csv",
        help="Destination path for per-image results CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "orb_development_summary.json",
        help="Destination path for JSON summary metrics.",
    )
    parser.add_argument(
        "--boxes-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "orb_development_boxes",
        help="Directory where per-image predicted boxes will be saved as .npy files.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="Provisional IoU threshold for one-to-one matching (default: 0.50).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads for parallel image evaluation (default: 4).",
    )
    args = parser.parse_args()

    rows = load_development_rows(args.manifest)
    print(f"Loaded {len(rows)} development records from {args.manifest}")
    results = evaluate_records(
        rows,
        iou_threshold=args.iou_threshold,
        workers=args.workers,
        boxes_directory=args.boxes_dir,
    )
    summary = summarise(results, iou_threshold=args.iou_threshold)
    write_results(results, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote results to {args.output}")
    print(f"Wrote summary to {args.summary}")
    print(
        f"Summary: {summary['successful']}/{summary['records']} successful, "
        f"Precision: {summary['overall_box_metrics']['precision']:.4f}, "
        f"Recall: {summary['overall_box_metrics']['recall']:.4f}, "
        f"F1: {summary['overall_box_metrics']['f1_score']:.4f}, "
        f"Mean runtime: {summary['runtime_ms']['mean']:.1f}ms"
    )
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
