"""Run comprehensive Enhanced Otsu validation grid search on 139 validation images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image, preprocess_pair
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.preprocessing import build_preprocessing_config


def extract_enhanced_otsu_boxes(
    diff_image: np.ndarray,
    blur_ksize: int = 3,
    morph_open: int = 3,
    morph_dilate: int = 25,
    min_area: float = 100.0,
) -> list[dict[str, int | float]]:
    diff_proc = diff_image
    if blur_ksize > 1:
        diff_proc = cv2.medianBlur(diff_image, blur_ksize)

    _, mask = cv2.threshold(
        diff_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if morph_open > 0:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_open, morph_open)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    if morph_dilate > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_dilate, morph_dilate)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes: list[dict[str, int | float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append(
            {
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + w),
                "ymax": int(y + h),
                "contour_area": area,
            }
        )
    return boxes


def run_otsu_grid_search(workers: int = 4) -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows for Otsu tuning.", flush=True)

    with open(PROJECT_ROOT / "configs" / "frozen_parameters.yaml") as f:
        cfg = yaml.safe_load(f)
    prep_cfg = build_preprocessing_config(cfg.get("preprocessing"))

    print("Preloading and computing difference images...", flush=True)
    cached_diffs = []
    for r in val_rows:
        ref = load_image(PROJECT_ROOT / r["reference_path"])
        def_img = load_image(PROJECT_ROOT / r["image_path"])
        ref_g, def_g = preprocess_pair(ref, def_img, prep_cfg)
        diff = cv2.absdiff(ref_g, def_g)
        gt_boxes = parse_voc_boxes(PROJECT_ROOT / r["annotation_path"])
        cached_diffs.append(
            {
                "image_id": r["image_id"],
                "defect_class": r["defect_class"],
                "diff": diff,
                "gt_boxes": gt_boxes,
            }
        )

    blur_options = [0, 3]
    open_options = [0, 3]
    dilate_options = [0, 15, 25, 35]
    min_area_options = [0.0, 50.0, 150.0, 300.0]

    all_results = []
    print("\n---> Running Otsu enhancement grid search across all combinations...", flush=True)
    t0 = perf_counter()

    for blur_k in blur_options:
        for open_k in open_options:
            for dilate_k in dilate_options:
                for min_a in min_area_options:
                    tp_tot = fp_tot = fn_tot = 0
                    runtimes = []
                    for item in cached_diffs:
                        t_start = perf_counter()
                        pred_boxes = extract_enhanced_otsu_boxes(
                            item["diff"],
                            blur_ksize=blur_k,
                            morph_open=open_k,
                            morph_dilate=dilate_k,
                            min_area=min_a,
                        )
                        runtimes.append((perf_counter() - t_start) * 1000.0)
                        eval_res = evaluate_boxes(
                            pred_boxes, item["gt_boxes"], iou_threshold=0.50
                        )
                        tp_tot += int(eval_res["true_positives"])
                        fp_tot += int(eval_res["false_positives"])
                        fn_tot += int(eval_res["false_negatives"])

                    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
                    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                    mean_rt = sum(runtimes) / len(runtimes)

                    all_results.append(
                        {
                            "blur_ksize": blur_k,
                            "morph_open": open_k,
                            "morph_dilate": dilate_k,
                            "min_area": min_a,
                            "true_positives": tp_tot,
                            "false_positives": fp_tot,
                            "false_negatives": fn_tot,
                            "precision": prec,
                            "recall": rec,
                            "f1_score": f1,
                            "mean_runtime_ms": mean_rt,
                        }
                    )

    print(f"Grid search completed in {perf_counter() - t0:.1f}s.", flush=True)
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "otsu_enhanced_grid_search.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}", flush=True)

    print("\n================ TOP 10 OTSU COMBINATIONS ON VALIDATION ================")
    print(results_df.head(10).to_string(index=False), flush=True)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_otsu_grid_search(workers=args.workers)
