"""Compare enhanced candidates on validation and freeze each F1 winner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image, preprocess_pair
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.preprocessing import build_preprocessing_config
from algorithms.otsu import detect_otsu
from algorithms.template_matching import detect_template_matching
from algorithms.canny import detect_canny
from algorithms.orb import detect_orb


ALGORITHM_CONFIG_KEYS = {
    "Otsu": "otsu",
    "Canny": "canny",
    "Template Matching": "template_matching",
    "ORB": "orb",
}


def freeze_f1_winners(
    results: pd.DataFrame,
    validation_protocol: dict,
    output_path: Path,
) -> dict:
    """Write one canonical test configuration from validation F1 winners."""
    expected = set(ALGORITHM_CONFIG_KEYS)
    present = set(results["algorithm"])
    if present != expected:
        raise ValueError(
            f"Cannot freeze incomplete validation results; expected {expected}, got {present}"
        )

    winners = (
        results.sort_values(
            ["algorithm", "f1_score", "combination_id"],
            ascending=[True, False, True],
        )
        .groupby("algorithm", sort=False, as_index=False)
        .first()
    )
    protocol = dict(validation_protocol["protocol"])
    protocol.pop("validation_images", None)
    selection_metric = protocol.pop("selection_metric")
    frozen = {
        "preprocessing": validation_protocol["preprocessing"],
        "protocol": protocol,
        "selection": {
            "split": "validation",
            "metric": selection_metric,
            "source": "outputs/metrics/essential_validation_comparison.csv",
            "winners": {},
        },
    }
    for row in winners.to_dict(orient="records"):
        algorithm = row["algorithm"]
        frozen[ALGORITHM_CONFIG_KEYS[algorithm]] = json.loads(row["parameter_config"])
        frozen["selection"]["winners"][ALGORITHM_CONFIG_KEYS[algorithm]] = {
            "combination_id": row["combination_id"],
            "f1_score": float(row["f1_score"]),
        }

    output_path.write_text(
        "# Generated from validation results. Do not edit using test results.\n"
        + yaml.safe_dump(frozen, sort_keys=False),
        encoding="utf-8",
    )
    return frozen


def main():
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows.")

    protocol_path = PROJECT_ROOT / "configs" / "validation_protocol.yaml"
    with protocol_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected_validation_count = int(cfg["protocol"]["validation_images"])
    if len(val_rows) != expected_validation_count:
        raise ValueError(
            f"Expected {expected_validation_count} validation rows, found {len(val_rows)}"
        )
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
            "description": "Validation reference: shared preprocessing + 64x64 window; no contour grouping",
            "parameters": {"method": "TM_CCOEFF_NORMED", "block_size": [64, 64], "step_size": 32, "corr_threshold": 0.60},
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(64, 64), step_size=32, corr_threshold=0.60, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Macro-96x96",
            "description": "Macro 96x96 sliding window with findContours",
            "parameters": {"method": "TM_CCOEFF_NORMED", "block_size": [96, 96], "step_size": 48, "corr_threshold": 0.60},
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(96, 96), step_size=48, corr_threshold=0.60, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Enhanced-32x32-Thresh55",
            "description": "Tight 32x32 window (thresh=0.55, high precision)",
            "parameters": {"method": "TM_CCOEFF_NORMED", "block_size": [32, 32], "step_size": 16, "corr_threshold": 0.55},
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(32, 32), step_size=16, corr_threshold=0.55, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "Template Matching",
            "combination_id": "TM-Best-32x32-Thresh65",
            "description": "Optimal 32x32 window (thresh=0.65, high recall)",
            "parameters": {"method": "TM_CCOEFF_NORMED", "block_size": [32, 32], "step_size": 16, "corr_threshold": 0.65},
            "run_fn": lambda ref, def_img: detect_template_matching(
                ref, def_img, block_size=(32, 32), step_size=16, corr_threshold=0.65, preprocessing_config=prep_cfg
            ),
        },
        # --- 2. Otsu's Thresholding ---
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Baseline",
            "description": "Validation reference: shared preprocessing; no Otsu-specific median blur, dilation, or area filter",
            "parameters": {"method": "THRESH_BINARY+THRESH_OTSU", "blur_ksize": 0, "morph_open": 0, "morph_dilate": 0, "min_area": 0.0},
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=0, morph_open=0, morph_dilate=0, min_area=0.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Filtered-NoDilate",
            "description": "Median blur + area filter (no dilation, micro IoU)",
            "parameters": {"method": "THRESH_BINARY+THRESH_OTSU", "blur_ksize": 3, "morph_open": 0, "morph_dilate": 0, "min_area": 150.0},
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=0, min_area=150.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Dilated-25",
            "description": "Median blur + dilation 25 + min_area 150",
            "parameters": {"method": "THRESH_BINARY+THRESH_OTSU", "blur_ksize": 3, "morph_open": 0, "morph_dilate": 25, "min_area": 150.0},
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=25, min_area=150.0
            ),
        },
        {
            "algorithm": "Otsu",
            "combination_id": "Otsu-Best-Dilated-35",
            "description": "Optimal median blur + dilation 35 + min_area 150",
            "parameters": {"method": "THRESH_BINARY+THRESH_OTSU", "blur_ksize": 3, "morph_open": 0, "morph_dilate": 35, "min_area": 150.0},
            "run_fn": lambda ref, def_img: detect_otsu(
                ref, def_img, preprocessing_config=prep_cfg, blur_ksize=3, morph_open=0, morph_dilate=35, min_area=150.0
            ),
        },
        # --- 3. Canny Edge Detection ---
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Baseline",
            "description": "Validation reference: shared preprocessing + low=50, high=150; no Canny-specific morphology",
            "parameters": {"low_threshold": 50, "high_threshold": 150, "aperture_size": 3, "l2_gradient": False, "morph_dilate": 0, "morph_close": 0, "min_area": 0.0},
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=0, min_area=0.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Closed-Area150",
            "description": "Edge difference with closing 5 + min_area 150",
            "parameters": {"low_threshold": 50, "high_threshold": 150, "aperture_size": 3, "l2_gradient": False, "morph_dilate": 0, "morph_close": 5, "min_area": 150.0},
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=150.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-LowThresh-Area300",
            "description": "Sensitive low=30, high=100 with closing 5 + min_area 300",
            "parameters": {"low_threshold": 30, "high_threshold": 100, "aperture_size": 3, "l2_gradient": False, "morph_dilate": 0, "morph_close": 5, "min_area": 300.0},
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=30.0, high_threshold=100.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=300.0
            ),
        },
        {
            "algorithm": "Canny",
            "combination_id": "Canny-Best-Area300",
            "description": "Optimal low=50, high=150 with closing 5 + min_area 300",
            "parameters": {"low_threshold": 50, "high_threshold": 150, "aperture_size": 3, "l2_gradient": False, "morph_dilate": 0, "morph_close": 5, "min_area": 300.0},
            "run_fn": lambda ref, def_img: detect_canny(
                ref, def_img, low_threshold=50.0, high_threshold=150.0, preprocessing_config=prep_cfg, morph_dilate=0, morph_close=5, min_area=300.0
            ),
        },
        # --- 4. ORB Feature Matching ---
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Baseline",
            "description": "Validation reference: shared preprocessing + discrete keypoint boxes (Hamming=60, R=35, Merge=False)",
            "parameters": {"n_features": 5000, "scale_factor": 1.2, "n_levels": 8, "matcher_type": "bf_crosscheck", "spatial_distance_threshold": 15.0, "hamming_threshold": 60.0, "box_radius": 35, "merge_points": False, "morph_dilate": 0, "min_area": 0.0},
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=60.0, box_radius=35, merge_points=False, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Merged-R35",
            "description": "Mask-clustered keypoint contours (Hamming=60, R=35, Merge=True)",
            "parameters": {"n_features": 5000, "scale_factor": 1.2, "n_levels": 8, "matcher_type": "bf_crosscheck", "spatial_distance_threshold": 15.0, "hamming_threshold": 60.0, "box_radius": 35, "merge_points": True, "morph_dilate": 0, "min_area": 0.0},
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=60.0, box_radius=35, merge_points=True, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Merged-R20",
            "description": "Compact mask-clustered contours (Hamming=40, R=20, Merge=True)",
            "parameters": {"n_features": 5000, "scale_factor": 1.2, "n_levels": 8, "matcher_type": "bf_crosscheck", "spatial_distance_threshold": 15.0, "hamming_threshold": 40.0, "box_radius": 20, "merge_points": True, "morph_dilate": 0, "min_area": 0.0},
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=40.0, box_radius=20, merge_points=True, preprocessing_config=prep_cfg
            ),
        },
        {
            "algorithm": "ORB",
            "combination_id": "ORB-Best-Area300",
            "description": "Optimal compact clustering with noise filter (Hamming=40, R=20, MinArea=300)",
            "parameters": {"n_features": 5000, "scale_factor": 1.2, "n_levels": 8, "matcher_type": "bf_crosscheck", "spatial_distance_threshold": 15.0, "hamming_threshold": 40.0, "box_radius": 20, "merge_points": True, "morph_dilate": 0, "min_area": 300.0},
            "run_fn": lambda ref, def_img: detect_orb(
                ref, def_img, hamming_threshold=40.0, box_radius=20, merge_points=True, min_area=300.0, preprocessing_config=prep_cfg
            ),
        },
    ]

    all_results = []
    print(f"\nRunning {len(experiments)} enhanced-candidate combinations on 139 validation images...")

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
                "parameter_config": json.dumps(exp["parameters"], separators=(",", ":")),
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
    frozen_path = PROJECT_ROOT / "configs" / "frozen_parameters.yaml"
    frozen = freeze_f1_winners(df, cfg, frozen_path)
    print(f"Generated canonical frozen configuration: {frozen_path}", flush=True)
    print(json.dumps(frozen["selection"]["winners"], indent=2), flush=True)


if __name__ == "__main__":
    main()
