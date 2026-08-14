"""Automated parameter tuning / grid search for ORB on the validation split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image, preprocess_pair
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.orb import extract_raw_boxes_from_points

VALIDATION_COUNT = 139


def load_validation_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "validation"
        ]
    if len(rows) != VALIDATION_COUNT:
        raise ValueError(
            f"Expected {VALIDATION_COUNT} validation records, found {len(rows)}"
        )
    return sorted(rows, key=lambda row: row["image_id"])


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_pair_features(
    row: dict[str, str], n_features: int = 5000
) -> dict[str, object]:
    """Precompute and cache ORB keypoints, descriptors, and raw matches once."""
    reference = load_image(resolve_project_path(row["reference_path"]))
    defective = load_image(resolve_project_path(row["image_path"]))
    gt_boxes = parse_voc_boxes(resolve_project_path(row["annotation_path"]))

    ref_gray, def_gray = preprocess_pair(reference, defective)
    image_shape = def_gray.shape[:2]

    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_def, des_def = orb.detectAndCompute(def_gray, None)

    bf_matches_data = []
    knn_matches_data = []
    all_def_coords = []

    if kp_def is not None and len(kp_def) > 0:
        all_def_coords = [(float(kp.pt[0]), float(kp.pt[1])) for kp in kp_def]

    has_descriptors = (
        des_ref is not None
        and des_def is not None
        and len(des_ref) > 0
        and len(des_def) > 0
    )

    if has_descriptors:
        # 1. BF CrossCheck
        bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cross_matches = bf_cross.match(des_ref, des_def)
        for m in cross_matches:
            pt_ref = kp_ref[m.queryIdx].pt
            pt_def = kp_def[m.trainIdx].pt
            disp = float(np.hypot(pt_def[0] - pt_ref[0], pt_def[1] - pt_ref[1]))
            bf_matches_data.append((m.trainIdx, float(pt_def[0]), float(pt_def[1]), disp, float(m.distance)))

        # 2. KNN Ratio
        bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = bf_knn.knnMatch(des_def, des_ref, k=2)
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                pt_def = kp_def[m.queryIdx].pt
                pt_ref = kp_ref[m.trainIdx].pt
                disp = float(np.hypot(pt_def[0] - pt_ref[0], pt_def[1] - pt_ref[1]))
                ratio = float(m.distance) / float(n.distance) if n.distance > 0 else 1.0
                knn_matches_data.append((m.queryIdx, float(pt_def[0]), float(pt_def[1]), disp, float(m.distance), ratio))

    return {
        "image_id": row["image_id"],
        "defect_class": row["defect_class"],
        "image_shape": image_shape,
        "gt_boxes": gt_boxes,
        "all_def_coords": all_def_coords,
        "bf_matches_data": bf_matches_data,
        "knn_matches_data": knn_matches_data,
        "num_def_kp": len(all_def_coords),
    }


def evaluate_candidate(
    cached_dataset: list[dict[str, object]],
    spatial_distance_threshold: float,
    hamming_threshold: float,
    box_radius: int,
    matcher_type: str,
    iou_threshold: float = 0.50,
) -> dict[str, float | int | str]:
    """Evaluate one parameter configuration in milliseconds using cached match vectors."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []

    for item in cached_dataset:
        image_shape = item["image_shape"]  # type: ignore[assignment]
        gt_boxes = item["gt_boxes"]  # type: ignore[assignment]
        all_def_coords = item["all_def_coords"]  # type: ignore[assignment]
        num_def_kp = item["num_def_kp"]  # type: ignore[assignment]

        defect_points: list[tuple[float, float]] = []

        if matcher_type == "bf_crosscheck":
            bf_data = item["bf_matches_data"]  # type: ignore[assignment]
            matched_indices = set()
            for train_idx, x, y, disp, dist in bf_data:
                matched_indices.add(train_idx)
                if disp > spatial_distance_threshold or dist > hamming_threshold:
                    defect_points.append((x, y))

            for idx in range(num_def_kp):
                if idx not in matched_indices:
                    defect_points.append(all_def_coords[idx])

        elif matcher_type == "knn_ratio":
            knn_data = item["knn_matches_data"]  # type: ignore[assignment]
            matched_indices = set()
            for query_idx, x, y, disp, dist, ratio in knn_data:
                if ratio < 0.75:
                    matched_indices.add(query_idx)
                    if disp > spatial_distance_threshold or dist > hamming_threshold:
                        defect_points.append((x, y))

            for idx in range(num_def_kp):
                if idx not in matched_indices:
                    defect_points.append(all_def_coords[idx])

        boxes = extract_raw_boxes_from_points(
            defect_points,
            image_shape=image_shape,
            box_radius=box_radius,
        )

        metrics = evaluate_boxes(boxes, gt_boxes, iou_threshold=iou_threshold)
        total_tp += int(metrics["true_positives"])
        total_fp += int(metrics["false_positives"])
        total_fn += int(metrics["false_negatives"])
        if metrics["mean_matched_iou"] > 0:
            matched_ious.append(float(metrics["mean_matched_iou"]))

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "matcher_type": matcher_type,
        "spatial_distance_threshold": spatial_distance_threshold,
        "hamming_threshold": hamming_threshold,
        "box_radius": box_radius,
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_matched_iou": mean_iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset_split.csv",
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "orb_validation_grid_search.csv",
        help="Path to save grid search results CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker threads for precomputing features.",
    )
    args = parser.parse_args()

    rows = load_validation_rows(args.manifest)
    print(f"Loaded {len(rows)} validation records from {args.manifest}")
    print("Precomputing ORB features & match descriptors for all validation images...")

    start_precompute = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        cached_dataset = list(
            executor.map(lambda r: extract_pair_features(r, n_features=5000), rows)
        )
    print(
        f"Precomputed {len(cached_dataset)} image pairs in {perf_counter() - start_precompute:.2f}s"
    )

    # Candidate search space
    spatial_candidates = [5.0, 10.0, 15.0, 20.0, 30.0]
    hamming_candidates = [30.0, 40.0, 45.0, 50.0, 60.0]
    box_radius_candidates = [15, 25, 35, 50]
    matcher_candidates = ["bf_crosscheck", "knn_ratio"]

    combinations = list(
        itertools.product(
            spatial_candidates,
            hamming_candidates,
            box_radius_candidates,
            matcher_candidates,
        )
    )

    print(f"Running grid search across {len(combinations)} parameter combinations...")
    start_search = perf_counter()

    results: list[dict[str, float | int | str]] = []
    for s_dist, h_dist, radius, matcher in combinations:
        res = evaluate_candidate(
            cached_dataset,
            spatial_distance_threshold=s_dist,
            hamming_threshold=h_dist,
            box_radius=radius,
            matcher_type=matcher,
            iou_threshold=0.50,
        )
        results.append(res)

    print(f"Completed grid search in {perf_counter() - start_search:.2f}s")

    # Sort results by F1-score (descending), then Recall (descending), then Precision
    results.sort(
        key=lambda x: (
            float(x["f1_score"]),
            float(x["recall"]),
            float(x["precision"]),
        ),
        reverse=True,
    )

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(results[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote full grid search results to {args.output}\n")
    print("=" * 95)
    print(f"{'RANK':<5} {'MATCHER':<14} {'SPATIAL':<8} {'HAMMING':<8} {'RADIUS':<7} {'F1-SCORE':<10} {'RECALL':<8} {'PRECISION':<10} {'TP':<5} {'FP':<8}")
    print("=" * 95)
    for rank, res in enumerate(results[:10], start=1):
        print(
            f"{rank:<5} {str(res['matcher_type']):<14} {res['spatial_distance_threshold']:<8.1f} {res['hamming_threshold']:<8.1f} {res['box_radius']:<7} {res['f1_score']:<10.6f} {res['recall']:<8.4f} {res['precision']:<10.6f} {res['true_positives']:<5} {res['false_positives']:<8}"
        )
    print("=" * 95)

    best = results[0]
    print(f"\nWINNING CONFIGURATION (Rank 1):")
    print(f"  Matcher:                    {best['matcher_type']}")
    print(f"  Spatial Distance Threshold: {best['spatial_distance_threshold']} px")
    print(f"  Hamming Threshold:          {best['hamming_threshold']}")
    print(f"  Box Radius:                 {best['box_radius']} px")
    print(f"  Validation F1-Score:        {best['f1_score']:.6f}")
    print(f"  Validation Recall:          {best['recall']:.4f}")
    print(f"  Validation Precision:       {best['precision']:.6f}")

    # Save winning frozen config to configs/orb_frozen_parameters.json
    config_path = PROJECT_ROOT / "configs" / "orb_frozen_parameters.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"Saved frozen configuration to {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
