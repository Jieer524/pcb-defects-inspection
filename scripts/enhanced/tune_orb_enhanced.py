"""Run comprehensive Enhanced ORB validation grid search on 139 validation images."""

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

from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.preprocessing import build_preprocessing_config
from algorithms.orb import detect_orb


def run_orb_grid_search() -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows for ORB tuning.", flush=True)

    with open(PROJECT_ROOT / "configs" / "frozen_parameters.yaml") as f:
        cfg = yaml.safe_load(f)
    prep_cfg = build_preprocessing_config(cfg.get("preprocessing"))
    orb_cfg = cfg.get("orb", {})

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
    box_radius_options = [20, 35, 50]
    merge_options = [False, True]
    morph_dilate_options = [0, 15, 25]
    min_area_options = [0.0, 300.0, 1000.0]
    hamming_options = [40.0, 60.0, 80.0]

    all_results = []
    print("\n---> Running ORB enhancement grid search...", flush=True)
    t0 = perf_counter()

    for hamming_th in hamming_options:
        for box_r in box_radius_options:
            for merge in merge_options:
                dilate_list = morph_dilate_options if merge else [0]
                area_list = min_area_options if merge else [0.0]
                for morph_d in dilate_list:
                    for min_a in area_list:
                        tp_tot = fp_tot = fn_tot = 0
                        runtimes = []
                        for item in cached_data:
                            t_start = perf_counter()
                            det = detect_orb(
                                item["ref"],
                                item["def_img"],
                                n_features=int(orb_cfg.get("n_features", 5000)),
                                scale_factor=float(orb_cfg.get("scale_factor", 1.2)),
                                n_levels=int(orb_cfg.get("n_levels", 8)),
                                matcher_type=str(orb_cfg.get("matcher_type", "bf_crosscheck")),
                                spatial_distance_threshold=float(orb_cfg.get("spatial_distance_threshold", 15.0)),
                                hamming_threshold=hamming_th,
                                box_radius=box_r,
                                calibrate=bool(orb_cfg.get("calibrate", True)),
                                ratio_threshold=float(orb_cfg.get("ratio_threshold", 0.75)),
                                ransac_reproj_threshold=float(orb_cfg.get("ransac_reproj_threshold", 5.0)),
                                preprocessing_config=prep_cfg,
                                merge_points=merge,
                                morph_dilate=morph_d,
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
                                "hamming_threshold": hamming_th,
                                "box_radius": box_r,
                                "merge_points": merge,
                                "morph_dilate": morph_d,
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
                        label = f"Ham={hamming_th}, R={box_r}, Merge={merge}, Dil={morph_d}, MinA={min_a}"
                        print(f"  {label} -> F1={f1:.4f} (TP={tp_tot}, FP={fp_tot}, FN={fn_tot})", flush=True)

    elapsed = perf_counter() - t0
    print(f"\nGrid search completed in {elapsed:.1f}s.", flush=True)
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "orb_enhanced_grid_search.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}", flush=True)

    print("\n================ TOP 10 ORB COMBINATIONS ON VALIDATION ================")
    print(results_df.head(10).to_string(index=False), flush=True)
    return results_df


if __name__ == "__main__":
    run_orb_grid_search()
