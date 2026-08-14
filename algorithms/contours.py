"""Shared direct-contour extraction for raw detection baselines."""

from __future__ import annotations

import cv2
import numpy as np


def extract_external_boxes(binary_map: np.ndarray) -> list[dict[str, int | float]]:
    """Return one box per external contour without filtering or merging."""
    if binary_map is None or binary_map.ndim != 2:
        raise ValueError("A two-dimensional binary map is required.")

    contours, _ = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes: list[dict[str, int | float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        boxes.append(
            {
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + width),
                "ymax": int(y + height),
                "contour_area": float(cv2.contourArea(contour)),
            }
        )
    return sorted(
        boxes,
        key=lambda box: (
            int(box["ymin"]),
            int(box["xmin"]),
            int(box["ymax"]),
            int(box["xmax"]),
        ),
    )
