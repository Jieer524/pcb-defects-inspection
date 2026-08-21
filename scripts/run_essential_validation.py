"""Run essential validation comparisons (4 combinations each) for all 4 algorithms on 139 validation images."""

from __future__ import annotations

import csv
import sys
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
from algorithms.otsu import detect_otsu
from algorithms.template_matching import detect_template_matching
from algorithms.canny import detect_canny
from algorithms.orb import detect_orb


def main():
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows.")

    with open(PROJECT_ROOT / "configs" / "frozen_parameters.yaml") as f:
        cfg = yaml.safe_load(f)
    prep_cfg = build_preprocessing_config(cfg.get("preprocessing"))

    print("Preloading validation image pairs...")
    cached_data = []
    for r in val_rows:
        ref = load_image(PROJECT_ROOT / r["reference_path"])
        def_img = load_image(PROJECT_ROOT / r["image_path"])
        gt_boxes = parse_voc_boxes(PROJECT_ROOT / r["annotation_path"])
        cached_data.append({"image_id": r["image_id"], "ref": ref, "def_img": def_img, "gt_boxes": gt_boxes})

    # Define the 4 essential combinations for each algorithm
    experiments = [
        # --- 1. Template Matching ---
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Baseline",
            "description": "Raw 64x64 baseline without contour merging",
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(64, 64), step_size=32, corr_threshold=0.60, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Macro-96x96",
            "description": "Macro 96x96 sliding window with findContours",
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(96, 96), step_size=48, corr_threshold=0.60, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Enhanced-32x32-Thresh55",
            "description": "Tight 32x32 window (thresh=0.55, high precision)",
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(32, 32), step_size=16, corr_threshold=0.55, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Best-32x32-Thresh65",
            "description": "Optimal 32x32 window (thresh=0.65, high recall)",
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(32, 32), step_size=16, corr_threshold=0.65, preprocessing_config=prep_cfg
            ),
        },
        # --- 2. Otsu's Thresholding ---
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Baseline",
            "description": "Raw Otsu difference without denoising or dilation",
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=0, morph_open=0, morph_dilate=0, min_area=0.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Filtered-NoDilate",
            "description": "Median blur + area filter (no dilation, micro IoU)",
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=0, min_area=150.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Dilated-25",
            "description": "Median blur + dilation 25 + min_area 150",
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=25, min_area=150.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Best-Dilated-35",
            "description": "Optimal median blur + dilation 35 + min_area 150",
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=35, min_area=150.0
            ),
        },
        # --- 3. Canny Edge Detection ---
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Baseline",
            "description": "Raw edge difference (low=50, high=150, no morph)",
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=0, min_area=0.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Closed-Area150",
            "description": "Edge difference with closing 5 + min_area 150",
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=150.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-LowThresh-Area300",
            "description": "Sensitive low=30, high=100 with closing 5 + min_area 300",
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=30.0, high_threshold=100.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=300.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Best-Area300",
            "description": "Optimal low=50, high=150 with closing 5 + min_area 300",
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=300.0
            ),
        },
        # --- 4. ORB Feature Matching ---
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Baseline",
            "description": "Raw discrete keypoint boxes (Hamming=60, R=35, Merge=False)",
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=60.0, box_radius=35, merge_points=False, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Merged-R35",
            "description": "Mask-clustered keypoint contours (Hamming=60, R=35, Merge=True)",
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=60.0, box_radius=35, merge_points=True, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Merged-R20",
            "description": "Compact mask-clustered contours (Hamming=40, R=20, Merge=True)",
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=40.0, box_radius=20, merge_points=True, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Best-Area300",
            "description": "Optimal compact clustering with noise filter (Hamming=40, R=20, MinArea=300)",
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=40.0, box_radius=20, merge_points=True, min_area=300.0, preprocessing_config=prep_cfg
            ),
        },
    ]

    all_results = []
    print("\nRunning 16 essential evaluation combinations on 139 validation images...")

    for exp in experiments:
        t0 = perf_counter()
        tp_tot = fp_tot = fn_tot = 0
        runtimes = []
        for item in cached_data:
            t_start = perf_counter()
            det = exp["run_fn"](item["ref"], item["def_img"])
            runtimes.append((perf_counter() - t_start) * 1000.0)
            eval_res = evaluate_boxes(det.boxes, item["gt_boxes"], iou_threshold=0.50)
            tp_tot += int(eval_res["true_positives"])
            fp_tot += int(eval_res["false_positives"])
            fn_tot += int(eval_res["false_negatives"])

        prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
        rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        mean_rt = sum(runtimes) / len(runtimes)

        all_results.append(
            {
                "algorithm": exp["algorithm"],
                "combination_id": exp["combination_id"],
                "description": exp["description"],
                "true_positives": tp_tot,
                "false_positives": fp_tot,
                "false_negatives": fn_tot,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "mean_runtime_ms": mean_rt,
            }
        )
        print(f"[{exp['algorithm']}] {exp['combination_id']:<28} -> F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, RT={mean_rt:.1f}ms", flush=True)

    df = pd.DataFrame(all_results)
    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "essential_validation_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved essential comparison table to: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
