"""Select raw Canny thresholds on the validation split without touching test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_canny_development import (
    DEFAULT_APERTURE_SIZE,
    DEFAULT_L2_GRADIENT,
    VALIDATION_CANDIDATE_PAIRS,
    evaluate_record,
)


VALIDATION_COUNT = 139


def load_validation_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if len(rows) != VALIDATION_COUNT:
        raise ValueError(f"Expected {VALIDATION_COUNT} validation rows, found {len(rows)}")
    return sorted(rows, key=lambda row: row["image_id"])


def evaluate_candidate(
    rows: list[dict[str, str]],
    low_threshold: int,
    high_threshold: int,
    iou_threshold: float,
    workers: int,
) -> dict[str, int | float | bool]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda row: evaluate_record(
                    row,
                    iou_threshold,
                    boxes_directory=None,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    aperture_size=DEFAULT_APERTURE_SIZE,
                    l2_gradient=DEFAULT_L2_GRADIENT,
                ),
                rows,
            )
        )
    errors = sum(result["status"] != "success" for result in results)
    if errors:
        raise RuntimeError(f"Canny candidate {(low_threshold, high_threshold)} had {errors} errors")
    tp = sum(int(result["true_positives"]) for result in results)
    fp = sum(int(result["false_positives"]) for result in results)
    fn = sum(int(result["false_negatives"]) for result in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "aperture_size": DEFAULT_APERTURE_SIZE,
        "l2_gradient": DEFAULT_L2_GRADIENT,
        "iou_threshold": iou_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_runtime_ms": sum(float(result["processing_time_ms"]) for result in results) / len(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "dataset_split.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "canny_validation_grid_search.csv",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--pairs",
        default=",".join(f"{low}:{high}" for low, high in VALIDATION_CANDIDATE_PAIRS),
        help="Comma-separated low:high pairs; defaults to the declared validation grid.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    candidate_pairs = []
    for pair_text in args.pairs.split(","):
        low_text, high_text = pair_text.split(":", maxsplit=1)
        pair = (int(low_text), int(high_text))
        if pair not in VALIDATION_CANDIDATE_PAIRS:
            raise ValueError(f"Undeclared validation candidate: {pair}")
        candidate_pairs.append(pair)

    rows = load_validation_rows(args.manifest)
    results = []
    for index, (low, high) in enumerate(candidate_pairs, start=1):
        candidate = evaluate_candidate(rows, low, high, args.iou_threshold, args.workers)
        results.append(candidate)
        print(
            f"Candidate {index}/{len(candidate_pairs)} "
            f"({low}, {high}): F1={candidate['f1_score']:.8f}",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)

    results.sort(
        key=lambda result: (
            float(result["f1_score"]),
            float(result["recall"]),
            float(result["precision"]),
            -float(result["mean_runtime_ms"]),
        ),
        reverse=True,
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    best = dict(results[0])
    best["selection_split"] = "validation"
    best["algorithm"] = "raw_canny"
    config_path = PROJECT_ROOT / "configs" / "canny_frozen_parameters.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Saved grid: {args.output}")
    print(f"Saved frozen config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
