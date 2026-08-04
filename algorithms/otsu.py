"""Raw Otsu defect detection without denoising or post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from algorithms.common import preprocess_pair


@dataclass(frozen=True)
class OtsuDetection:
    """Outputs from one raw Otsu comparison."""

    threshold: float
    reference_gray: np.ndarray
    defective_gray: np.ndarray
    difference: np.ndarray
    mask: np.ndarray
    boxes: list[dict[str, int | float]]
    processing_time_ms: float


def extract_raw_boxes(binary_mask: np.ndarray) -> list[dict[str, int | float]]:
    """Return one box per external contour without filtering or merging."""
    if binary_mask is None or binary_mask.ndim != 2:
        raise ValueError("A two-dimensional binary mask is required.")

    contours, _ = cv2.findContours(
        binary_mask,
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


def detect_otsu(reference: np.ndarray, defective: np.ndarray) -> OtsuDetection:
    """Run the complete raw Otsu pipeline on an aligned image pair.

    Timing includes validation, grayscale conversion, absolute difference, Otsu
    thresholding, direct contour extraction, and box construction. Disk I/O is
    deliberately excluded so algorithm timings remain comparable.
    """
    start_time = perf_counter()

    reference_gray, defective_gray = preprocess_pair(reference, defective)
    difference = cv2.absdiff(reference_gray, defective_gray)
    threshold, mask = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    boxes = extract_raw_boxes(mask)

    processing_time_ms = (perf_counter() - start_time) * 1000.0
    return OtsuDetection(
        threshold=float(threshold),
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        difference=difference,
        mask=mask,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
    )
