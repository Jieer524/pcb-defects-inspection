"""Run comprehensive Template Matching validation grid search with enhanced post-processing."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import load_image, preprocess_pair
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.preprocessing import build_preprocessing_config
from algorithms.template_matching import _window_positions


def compute_similarity_map(
    ref_gray: np.ndarray,
    def_gray: np.ndarray,
    block_size: tuple[int, int],
    step_size: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    height, width = ref_gray.shape
    b_h, b_w = block_size
    y_pos = _window_positions(height, b_h, step_size)
    x_pos = _window_positions(width, b_w, step_size)
    sim_map = np.empty((len(y_pos), len(x_pos)), dtype=np.float32)
    for r_idx, y in enumerate(y_pos):
        for c_idx, x in enumerate(x_pos):
            ref_patch = ref_gray[y : y + b_h, x : x + b_w]
            def_patch = def_gray[y : y + b_h, x : x + b_w]
            score = float(
                cv2.matchTemplate(def_patch, ref_patch, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            sim_map[r_idx, c_idx] = score
    return sim_map, y_pos, x_pos


def extract_enhanced_boxes(
    sim_map: np.ndarray,
    y_pos: list[int],
    x_pos: list[int],
    img_shape: tuple[int, int],
    block_size: tuple[int, int],
    corr_threshold: float,
    morph_kernel_size: int = 0,
    min_area: float = 0.0,
) -> list[dict[str, int | float]]:
    height, width = img_shape
    b_h, b_w = block_size
    defect_mask = np.zeros((height, width), dtype=np.uint8)

    for r_idx, c_idx in np.argwhere(sim_map < corr_threshold):
        y = y_pos[int(r_idx)]
        x = x_pos[int(c_idx)]
        defect_mask[y : y + b_h, x : x + b_w] = 255

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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


def run_grid_search(workers: int = 4) -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "dataset_split.csv")
    val_rows = manifest[manifest["split"] == "validation"].to_dict(orient="records")
    print(f"Loaded {len(val_rows)} validation rows.", flush=True)

    with open(PROJECT_ROOT / "configs" / "frozen_parameters.yaml") as f:
        cfg = yaml.safe_load(f)
    prep_cfg = build_preprocessing_config(cfg.get("preprocessing"))

    # Load and preprocess pairs once
    print("Preloading and preprocessing validation images...", flush=True)
    cached_pairs = []
    for r in val_rows:
        ref = load_image(PROJECT_ROOT / r["reference_path"])
        def_img = load_image(PROJECT_ROOT / r["image_path"])
        ref_g, def_g = preprocess_pair(ref, def_img, prep_cfg)
        gt_boxes = parse_voc_boxes(PROJECT_ROOT / r["annotation_path"])
        cached_pairs.append(
            {
                "image_id": r["image_id"],
                "defect_class": r["defect_class"],
                "ref_gray": ref_g,
                "def_gray": def_g,
                "gt_boxes": gt_boxes,
            }
        )

    grid_configurations = [
        {"block_size": (32, 32), "step_size": 16},
        {"block_size": (48, 48), "step_size": 24},
        {"block_size": (64, 64), "step_size": 32},
        {"block_size": (96, 96), "step_size": 48},
    ]
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    morph_options = [0, 5]
    min_area_options = [0.0, 300.0]

    all_results = []

    for grid_cfg in grid_configurations:
        b_size = grid_cfg["block_size"]
        s_size = grid_cfg["step_size"]
        print(f"\n---> Computing similarity maps for Block: {b_size}, Step: {s_size}...", flush=True)
        t0 = perf_counter()

        def compute_one(item):
            s_map, y_p, x_p = compute_similarity_map(
                item["ref_gray"], item["def_gray"], b_size, s_size
            )
            return {
                "image_id": item["image_id"],
                "defect_class": item["defect_class"],
                "shape": item["ref_gray"].shape,
                "sim_map": s_map,
                "y_pos": y_p,
                "x_pos": x_p,
                "gt_boxes": item["gt_boxes"],
            }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            val_sim_maps = list(executor.map(compute_one, cached_pairs))

        compute_dur = (perf_counter() - t0) * 1000.0 / len(val_sim_maps)
        print(f"     Done in {perf_counter() - t0:.1f}s (avg {compute_dur:.1f}ms/img). Testing threshold & filter combinations...", flush=True)

        for th in thresholds:
            for morph in morph_options:
                for min_a in min_area_options:
                    tp_tot = fp_tot = fn_tot = 0
                    for item in val_sim_maps:
                        pred_boxes = extract_enhanced_boxes(
                            item["sim_map"],
                            item["y_pos"],
                            item["x_pos"],
                            item["shape"],
                            b_size,
                            th,
                            morph_kernel_size=morph,
                            min_area=min_a,
                        )
                        eval_res = evaluate_boxes(
                            pred_boxes, item["gt_boxes"], iou_threshold=0.50
                        )
                        tp_tot += int(eval_res["true_positives"])
                        fp_tot += int(eval_res["false_positives"])
                        fn_tot += int(eval_res["false_negatives"])

                    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
                    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

                    all_results.append(
                        {
                            "block_size": f"{b_size[0]}x{b_size[1]}",
                            "step_size": s_size,
                            "corr_threshold": th,
                            "morph_close": morph,
                            "min_area": min_a,
                            "true_positives": tp_tot,
                            "false_positives": fp_tot,
                            "false_negatives": fn_tot,
                            "precision": prec,
                            "recall": rec,
                            "f1_score": f1,
                            "mean_runtime_ms": compute_dur,
                        }
                    )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "template_matching_enhanced_grid_search.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved full grid search results to: {out_csv}", flush=True)

    print("\n================ TOP 10 COMBINATIONS ON VALIDATION ================")
    print(results_df.head(10).to_string(index=False), flush=True)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_grid_search(workers=args.workers)
