"""Run raw Otsu on the manifest's development split only."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.otsu import detect_otsu


ALGORITHM_VERSION = "raw-otsu-v1"
DEVELOPMENT_COUNT = 139
RESULT_FIELDS = [
    "image_id",
    "defect_class",
    "split",
    "algorithm_version",
    "otsu_threshold",
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
    "white_pixel_percentage",
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
    """Cache the ten read-only reference images during a dataset run."""
    return load_image(path)


def evaluate_record(
    row: dict[str, str],
    iou_threshold: float,
    boxes_directory: Path | None = None,
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
        detection = detect_otsu(reference, defective)
        metrics = evaluate_boxes(
            detection.boxes,
            ground_truth_boxes,
            iou_threshold=iou_threshold,
        )
        white_pixel_percentage = (
            float((detection.mask != 0).sum()) / detection.mask.size * 100.0
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
                    ("contour_area", "<f4"),
                ]
            )
            compact_boxes = np.fromiter(
                (
                    (
                        box["xmin"],
                        box["ymin"],
                        box["xmax"],
                        box["ymax"],
                        box["contour_area"],
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
                "otsu_threshold": detection.threshold,
                "predicted_count": len(detection.boxes),
                "ground_truth_count": len(ground_truth_boxes),
                **metrics,
                "white_pixel_percentage": white_pixel_percentage,
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
) -> list[dict[str, str | int | float]]:
    """Evaluate independent records concurrently while preserving manifest order."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        results = []
        for index, row in enumerate(rows, start=1):
            results.append(evaluate_record(row, iou_threshold, boxes_directory))
            if index % 10 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda row: evaluate_record(row, iou_threshold, boxes_directory),
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
        "iou_threshold": iou_threshold,
        "predicted_boxes_format": (
            "NumPy structured array: xmin,ymin,xmax,ymax int32; contour_area float32"
        ),
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
                1000.0 / statistics.fmean(runtimes)
                if runtimes and statistics.fmean(runtimes)
                else 0.0
            ),
        },
    }


def write_results(
    results: list[dict[str, str | int | float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset_split.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "otsu_development.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "otsu_development_summary.json",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Provisional development diagnostic; freeze the final rule after validation.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = load_development_rows(args.manifest)
    boxes_directory = args.output.parent / "otsu_development_boxes"
    results = evaluate_records(
        rows,
        args.iou_threshold,
        workers=args.workers,
        boxes_directory=boxes_directory,
    )
    summary = summarise(results, args.iou_threshold)
    write_results(results, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results: {args.output}")
    print(f"Summary: {args.summary}")

    if int(summary["errors"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
