"""Run the four frozen raw algorithms once on the held-out test split."""

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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.canny import detect_canny
from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.orb import detect_orb
from algorithms.otsu import detect_otsu
from algorithms.preprocessing import build_preprocessing_config
from algorithms.template_matching import detect_template_matching

TEST_COUNT = 415
ALGORITHM_VERSION = "raw-frozen-v2"
ALGORITHMS = ("otsu", "canny", "template_matching", "orb")


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_test_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if len(rows) != TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} test records, found {len(rows)}")
    return sorted(rows, key=lambda row: row["image_id"])


def load_frozen_config(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["test_images"] != TEST_COUNT or protocol["iou_threshold"] != 0.5:
        raise ValueError("Frozen protocol must specify 415 test images and IoU 0.50.")
    return config


def resolve_preprocessing_config(config: dict[str, object]) -> dict:
    """Build the preprocessing_config dict consumed by the frozen algorithms."""
    return build_preprocessing_config(config.get("preprocessing"))


@lru_cache(maxsize=16)
def load_reference(path: Path) -> np.ndarray:
    return load_image(path)


def run_detector(
    algorithm: str,
    reference: np.ndarray,
    defective: np.ndarray,
    config: dict[str, object],
    preprocessing_config: dict,
):
    if algorithm == "otsu":
        params = config.get("otsu", {})
        return detect_otsu(
            reference,
            defective,
            blur_ksize=int(params.get("blur_ksize", 3)),
            morph_open=int(params.get("morph_open", 0)),
            morph_dilate=int(params.get("morph_dilate", 35)),
            min_area=float(params.get("min_area", 150.0)),
            preprocessing_config=preprocessing_config,
        )
    if algorithm == "canny":
        params = config.get("canny", {})
        return detect_canny(
            reference,
            defective,
            low_threshold=float(params["low_threshold"]),
            high_threshold=float(params["high_threshold"]),
            aperture_size=int(params["aperture_size"]),
            l2_gradient=bool(params["l2_gradient"]),
            morph_dilate=int(params.get("morph_dilate", 0)),
            morph_close=int(params.get("morph_close", 5)),
            min_area=float(params.get("min_area", 300.0)),
            preprocessing_config=preprocessing_config,
        )
    if algorithm == "template_matching":
        params = config.get("template_matching", {})
        return detect_template_matching(
            reference,
            defective,
            block_size=tuple(params["block_size"]),
            step_size=int(params["step_size"]),
            corr_threshold=float(params["corr_threshold"]),
            preprocessing_config=preprocessing_config,
        )
    if algorithm == "orb":
        params = config.get("orb", {})
        return detect_orb(
            reference,
            defective,
            n_features=int(params["n_features"]),
            scale_factor=float(params["scale_factor"]),
            n_levels=int(params["n_levels"]),
            spatial_distance_threshold=float(params["spatial_distance_threshold"]),
            hamming_threshold=float(params["hamming_threshold"]),
            box_radius=int(params["box_radius"]),
            matcher_type=str(params["matcher_type"]),
            calibrate=bool(params.get("calibrate", False)),
            ransac_reproj_threshold=float(params.get("ransac_reproj_threshold", 5.0)),
            merge_points=bool(params.get("merge_points", True)),
            morph_dilate=int(params.get("morph_dilate", 0)),
            min_area=float(params.get("min_area", 300.0)),
            preprocessing_config=preprocessing_config,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def save_boxes(boxes: list[dict[str, int | float]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(
        [[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in boxes],
        dtype=np.int32,
    ).reshape(-1, 4)
    np.save(path, coordinates, allow_pickle=False)
    return path.relative_to(PROJECT_ROOT).as_posix()


def evaluate_record(
    algorithm: str,
    row: dict[str, str],
    config: dict[str, object],
    boxes_root: Path,
    preprocessing_config: dict,
) -> dict[str, object]:
    result: dict[str, object] = {
        "image_id": row["image_id"],
        "defect_class": row["defect_class"],
        "split": row["split"],
        "algorithm": algorithm,
        "algorithm_version": ALGORITHM_VERSION,
        "parameter_config": json.dumps(
            {
                **{
                    name: config[name]
                    for name in ("otsu", "canny", "template_matching", "orb")
                    if name in config
                },
                "preprocessing": config.get("preprocessing", {}),
            },
            separators=(",", ":"),
        ),
        "status": "error",
        "error": "",
    }
    try:
        reference = load_reference(resolve_project_path(row["reference_path"]))
        defective = load_image(resolve_project_path(row["image_path"]))
        truth = parse_voc_boxes(resolve_project_path(row["annotation_path"]))
        detection = run_detector(
            algorithm, reference, defective, config, preprocessing_config
        )
        metrics = evaluate_boxes(
            detection.boxes,
            truth,
            iou_threshold=float(config["protocol"]["iou_threshold"]),
        )
        boxes_path = boxes_root / algorithm / f"{row['image_id']}.npy"
        result.update(
            {
                "predicted_count": len(detection.boxes),
                "ground_truth_count": len(truth),
                **metrics,
                "false_positives_per_image": metrics["false_positives"],
                "processing_time_ms": detection.processing_time_ms,
                "predicted_boxes_path": save_boxes(detection.boxes, boxes_path),
                "status": "success",
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row["status"] == "success"]
    tp = sum(int(row["true_positives"]) for row in successful)
    fp = sum(int(row["false_positives"]) for row in successful)
    fn = sum(int(row["false_negatives"]) for row in successful)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    runtimes = [float(row["processing_time_ms"]) for row in successful]
    return {
        "images": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": statistics.fmean(
            float(row["mean_matched_iou"]) for row in successful
        ) if successful else 0.0,
        "false_positives_per_image": fp / len(successful) if successful else 0.0,
        "mean_runtime_ms": statistics.fmean(runtimes) if runtimes else 0.0,
        "runtime_standard_deviation_ms": statistics.pstdev(runtimes) if runtimes else 0.0,
        "frames_per_second": 1000.0 / statistics.fmean(runtimes) if runtimes and statistics.fmean(runtimes) else 0.0,
    }


def summarise(results: list[dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    by_algorithm: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_algorithm_class: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_algorithm[str(row["algorithm"])].append(row)
        by_algorithm_class[(str(row["algorithm"]), str(row["defect_class"]))].append(row)
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "split": "test",
        "test_images": TEST_COUNT,
        "iou_threshold": config["protocol"]["iou_threshold"],
        "preprocessing": config.get("preprocessing", {}),
        "orb_calibrate": bool(config["orb"].get("calibrate", False)),
        "class_distribution": dict(sorted(Counter(row["defect_class"] for row in results if row["algorithm"] == "otsu").items())),
        "overall": {name: aggregate(by_algorithm[name]) for name in ALGORITHMS},
        "by_class": {
            name: {
                defect_class: aggregate(class_rows)
                for (algorithm, defect_class), class_rows in sorted(by_algorithm_class.items())
                if algorithm == name
            }
            for name in ALGORITHMS
        },
    }


def write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in results for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-frozen-test", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "dataset_split.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "frozen_parameters.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "metrics" / "final_test_results.csv")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "metrics" / "final_test_summary.json")
    args = parser.parse_args()
    if not args.confirm_frozen_test:
        parser.error("Pass --confirm-frozen-test only after every configuration is frozen.")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config = load_frozen_config(args.config)
    rows = load_test_rows(args.manifest)
    preprocessing_config = resolve_preprocessing_config(config)
    boxes_root = args.output.parent / "final_test_boxes"
    all_results: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        print(f"Running frozen {algorithm} on {len(rows)} test images...", flush=True)
        if args.workers == 1:
            algorithm_results = []
            for index, row in enumerate(rows, start=1):
                algorithm_results.append(
                    evaluate_record(algorithm, row, config, boxes_root, preprocessing_config)
                )
                if index % 10 == 0 or index == len(rows):
                    print(f"  {algorithm}: {index}/{len(rows)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                algorithm_results = []
                evaluated = executor.map(
                    lambda row: evaluate_record(
                        algorithm, row, config, boxes_root, preprocessing_config
                    ),
                    rows,
                )
                for index, result in enumerate(evaluated, start=1):
                    algorithm_results.append(result)
                    if index % 10 == 0 or index == len(rows):
                        print(f"  {algorithm}: {index}/{len(rows)}", flush=True)
        all_results.extend(algorithm_results)
        write_csv(all_results, args.output)
        print(f"Completed {algorithm}", flush=True)

    summary = summarise(all_results, config)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Results: {args.output}")
    print(f"Summary: {args.summary}")
    if any(values["errors"] for values in summary["overall"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
