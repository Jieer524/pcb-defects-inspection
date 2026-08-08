"""Shared annotation parsing and detection metrics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


Box = Mapping[str, int | float | str]


def parse_voc_boxes(annotation_path: str | Path) -> list[dict[str, int | str]]:
    """Parse Pascal VOC boxes, preserving the dataset's stored coordinates."""
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation not found: {path}")

    root = ET.parse(path).getroot()
    boxes: list[dict[str, int | str]] = []
    for obj in root.findall("object"):
        bounds = obj.find("bndbox")
        if bounds is None:
            raise ValueError(f"Object without bndbox in {path}")
        box = {
            "class": obj.findtext("name", default="unknown"),
            "xmin": int(bounds.findtext("xmin", default="-1")),
            "ymin": int(bounds.findtext("ymin", default="-1")),
            "xmax": int(bounds.findtext("xmax", default="-1")),
            "ymax": int(bounds.findtext("ymax", default="-1")),
        }
        if box["xmin"] >= box["xmax"] or box["ymin"] >= box["ymax"]:
            raise ValueError(f"Invalid box in {path}: {box}")
        boxes.append(box)
    return boxes


def box_iou(box_a: Box, box_b: Box) -> float:
    """Calculate intersection-over-union for two axis-aligned boxes."""
    intersection_width = max(
        0.0,
        min(float(box_a["xmax"]), float(box_b["xmax"]))
        - max(float(box_a["xmin"]), float(box_b["xmin"])),
    )
    intersection_height = max(
        0.0,
        min(float(box_a["ymax"]), float(box_b["ymax"]))
        - max(float(box_a["ymin"]), float(box_b["ymin"])),
    )
    intersection = intersection_width * intersection_height
    area_a = max(0.0, float(box_a["xmax"]) - float(box_a["xmin"])) * max(
        0.0, float(box_a["ymax"]) - float(box_a["ymin"])
    )
    area_b = max(0.0, float(box_b["xmax"]) - float(box_b["xmin"])) * max(
        0.0, float(box_b["ymax"]) - float(box_b["ymin"])
    )
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_boxes(
    predicted_boxes: Sequence[Box],
    ground_truth_boxes: Sequence[Box],
    iou_threshold: float = 0.5,
) -> dict[str, int | float]:
    """Greedily perform one-to-one box matching in descending IoU order."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")

    candidates: list[tuple[float, int, int]] = []
    if predicted_boxes and ground_truth_boxes:
        predicted = np.asarray(
            [
                [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
                for box in predicted_boxes
            ],
            dtype=np.float64,
        )
        predicted_areas = np.maximum(0.0, predicted[:, 2] - predicted[:, 0]) * np.maximum(
            0.0, predicted[:, 3] - predicted[:, 1]
        )
        for truth_index, truth in enumerate(ground_truth_boxes):
            truth_coordinates = np.asarray(
                [truth["xmin"], truth["ymin"], truth["xmax"], truth["ymax"]],
                dtype=np.float64,
            )
            intersection_widths = np.maximum(
                0.0,
                np.minimum(predicted[:, 2], truth_coordinates[2])
                - np.maximum(predicted[:, 0], truth_coordinates[0]),
            )
            intersection_heights = np.maximum(
                0.0,
                np.minimum(predicted[:, 3], truth_coordinates[3])
                - np.maximum(predicted[:, 1], truth_coordinates[1]),
            )
            intersections = intersection_widths * intersection_heights
            truth_area = max(0.0, truth_coordinates[2] - truth_coordinates[0]) * max(
                0.0, truth_coordinates[3] - truth_coordinates[1]
            )
            unions = predicted_areas + truth_area - intersections
            ious = np.divide(
                intersections,
                unions,
                out=np.zeros_like(intersections),
                where=unions != 0.0,
            )
            qualifying = np.flatnonzero(ious >= iou_threshold)
            candidates.extend(
                (float(ious[index]), int(index), truth_index) for index in qualifying
            )
    candidates.sort(reverse=True)
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    matched_ious: list[float] = []
    for iou, predicted_index, truth_index in candidates:
        if predicted_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(predicted_index)
        matched_truth.add(truth_index)
        matched_ious.append(iou)

    true_positives = len(matched_ious)
    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "iou_threshold": float(iou_threshold),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
    }
