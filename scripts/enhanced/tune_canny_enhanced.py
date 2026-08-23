"""Run comprehensive Enhanced Canny validation grid search on 139 validation images."""

from __future__ import annotations

import argparse
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
from algorithms.canny import detect_canny


def run_canny_grid_search() -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows for Canny tuning.", flush=True)

    with open(PROJECT_ROOT / "configs" / "frozen_parameters.yaml") as f:
        cfg = yaml.safe_load(f)
    prep_cfg = build_preprocessing_config(cfg.get("preprocessing"))

    print("Preloading validation images...", flush=True)
    cached_data = []
    for r in val_rows:
        ref = load_image(PROJECT_ROOT / r["reference_path"])
        def_img = load_image(PROJECT_ROOT / r["image_path"])
        gt_boxes = parse_voc_boxes(PROJECT_ROOT / r["annotation_path"])
        cached_data.append(
            {"image_id": r["image_id"], "ref": ref, "def_img": def_img, "gt_boxes": gt_boxes}
        )

    # Grid search parameters
    canny_configs = [
        {"low": 30, "high": 100},
        {"low": 50, "high": 150},
        {"low": 80, "high": 200},
    ]
    morph_dilate_options = [0, 15, 25, 35]
    morph_close_options = [0, 5]
    min_area_options = [0.0, 50.0, 150.0, 300.0]

    all_results = []
    print("\n---> Running Canny enhancement grid search...", flush=True)
    t0 = perf_counter()

    for cc in canny_configs:
        low, high = cc["low"], cc["high"]
        for morph_d in morph_dilate_options:
            for morph_c in morph_close_options:
                for min_a in min_area_options:
                    tp_tot = fp_tot = fn_tot = 0
                    runtimes = []
                    for item in cached_data:
                        t_start = perf_counter()
                        det = detect_canny(
                            item["ref"],
                            item["def_img"],
                            low_threshold=float(low),
                            high_threshold=float(high),
                            aperture_size=3,
                            l2_gradient=False,
                            preprocessing_config=prep_cfg,
                            morph_dilate=morph_d,
                            morph_close=morph_c,
                            min_area=min_a,
                        )
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
                            "low_threshold": low,
                            "high_threshold": high,
                            "morph_dilate": morph_d,
                            "morph_close": morph_c,
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

    elapsed = perf_counter() - t0
    print(f"Grid search completed in {elapsed:.1f}s.", flush=True)
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "canny_enhanced_grid_search.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}", flush=True)

    print("\n================ TOP 10 CANNY COMBINATIONS ON VALIDATION ================")
    print(results_df.head(10).to_string(index=False), flush=True)
    return results_df


if __name__ == "__main__":
    run_canny_grid_search()
