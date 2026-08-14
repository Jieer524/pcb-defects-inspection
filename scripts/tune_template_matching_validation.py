"""Select raw blockwise template-matching parameters on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.template_matching import (
    boxes_from_similarity_map,
    detect_template_matching,
)
from scripts.run_template_matching_development import VALIDATION_CANDIDATES


VALIDATION_COUNT = 139


def load_validation_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if len(rows) != VALIDATION_COUNT:
        raise ValueError(f"Expected {VALIDATION_COUNT} validation rows, found {len(rows)}")
    return sorted(rows, key=lambda row: row["image_id"])


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_similarity_record(
    row: dict[str, str], block_size: tuple[int, int], step_size: int
) -> dict[str, object]:
    reference = load_image(resolve_path(row["reference_path"]))
    defective = load_image(resolve_path(row["image_path"]))
    detection = detect_template_matching(
        reference,
        defective,
        block_size=block_size,
        step_size=step_size,
        corr_threshold=0.0,
    )
    return {
        "image_shape": detection.defective_gray.shape,
        "similarity_map": detection.similarity_map,
        "ground_truth_boxes": parse_voc_boxes(resolve_path(row["annotation_path"])),
        "similarity_time_ms": detection.processing_time_ms,
    }


def evaluate_threshold(
    cached_records: list[dict[str, object]],
    block_size: tuple[int, int],
    step_size: int,
    corr_threshold: float,
    iou_threshold: float,
) -> dict[str, int | float | str]:
    tp = fp = fn = 0
    total_time_ms = 0.0
    for record in cached_records:
        start = perf_counter()
        boxes = boxes_from_similarity_map(
            record["similarity_map"],  # type: ignore[arg-type]
            record["image_shape"],  # type: ignore[arg-type]
            block_size,
            step_size,
            corr_threshold,
        )
        metrics = evaluate_boxes(
            boxes,
            record["ground_truth_boxes"],  # type: ignore[arg-type]
            iou_threshold=iou_threshold,
        )
        total_time_ms += float(record["similarity_time_ms"])
        total_time_ms += (perf_counter() - start) * 1000.0
        tp += int(metrics["true_positives"])
        fp += int(metrics["false_positives"])
        fn += int(metrics["false_negatives"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "method": "TM_CCOEFF_NORMED",
        "block_height": block_size[0],
        "block_width": block_size[1],
        "step_size": step_size,
        "corr_threshold": corr_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_runtime_ms": total_time_ms / len(cached_records),
    }


def write_grid(results: list[dict[str, int | float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "dataset_split.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "template_matching_validation_grid_search.csv",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    rows = load_validation_rows(args.manifest)
    results: list[dict[str, int | float | str]] = []
    pairs = VALIDATION_CANDIDATES["block_step_pairs"]
    thresholds = VALIDATION_CANDIDATES["corr_threshold"]
    for pair_index, pair in enumerate(pairs, start=1):
        block_size = tuple(pair["block_size"])
        step_size = int(pair["step_size"])
        print(
            f"Computing validation similarities {pair_index}/{len(pairs)}: "
            f"block={block_size}, step={step_size}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            cached_records = list(
                executor.map(
                    lambda row: compute_similarity_record(row, block_size, step_size),
                    rows,
                )
            )
        for threshold in thresholds:
            candidate = evaluate_threshold(
                cached_records,
                block_size,
                step_size,
                float(threshold),
                args.iou_threshold,
            )
            results.append(candidate)
            print(
                f"  threshold={threshold:.2f}: F1={candidate['f1_score']:.6f}",
                flush=True,
            )
        write_grid(results, args.output)

    results.sort(
        key=lambda result: (
            float(result["f1_score"]),
            float(result["recall"]),
            float(result["precision"]),
            -float(result["mean_runtime_ms"]),
        ),
        reverse=True,
    )
    write_grid(results, args.output)
    best = dict(results[0])
    best["selection_split"] = "validation"
    best["algorithm"] = "raw_template_matching"
    best["selection_metric"] = "f1_score"
    config_path = PROJECT_ROOT / "configs" / "template_matching_frozen_parameters.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Saved grid: {args.output}")
    print(f"Saved frozen config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
