"""Raw Canny defect detection with Gaussian smoothing and no post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from algorithms.common import preprocess_pair


@dataclass(frozen=True)
class CannyDetection:
    """Outputs from one raw Canny comparison."""

    low_threshold: int
    high_threshold: int
    blur_kernel_size: int
    reference_gray: np.ndarray
    defective_gray: np.ndarray
    difference: np.ndarray
    blurred_difference: np.ndarray
    edge_map: np.ndarray
    boxes: list[dict[str, int | float]]
    processing_time_ms: float


def extract_raw_boxes(edge_map: np.ndarray) -> list[dict[str, int | float]]:
    """Return one box per external edge contour without filtering or merging."""
    if edge_map is None or edge_map.ndim != 2:
        raise ValueError("A two-dimensional edge map is required.")

    contours, _ = cv2.findContours(
        edge_map,
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


def detect_canny(
    reference: np.ndarray,
    defective: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel_size: int = 5,
) -> CannyDetection:
    """Run the complete raw Canny pipeline on an aligned image pair.

    Timing includes validation, grayscale conversion, absolute difference,
    Gaussian smoothing, Canny edge extraction, direct contour extraction,
    and box construction. Disk I/O is deliberately excluded so algorithm
    timings remain comparable.
    """
    if blur_kernel_size < 3 or blur_kernel_size % 2 == 0:
        raise ValueError(
            "blur_kernel_size must be an odd integer greater than or equal to 3."
        )

    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError(
            "Thresholds must satisfy "
            "0 <= low_threshold < high_threshold <= 255."
        )

    start_time = perf_counter()

    reference_gray, defective_gray = preprocess_pair(reference, defective)

    difference = cv2.absdiff(
        reference_gray,
        defective_gray,
    )

    blurred_difference = cv2.GaussianBlur(
        difference,
        (blur_kernel_size, blur_kernel_size),
        0,
    )

    edge_map = cv2.Canny(
        blurred_difference,
        low_threshold,
        high_threshold,
    )

    boxes = extract_raw_boxes(edge_map)

    processing_time_ms = (perf_counter() - start_time) * 1000.0

    return CannyDetection(
        low_threshold=int(low_threshold),
        high_threshold=int(high_threshold),
        blur_kernel_size=int(blur_kernel_size),
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        difference=difference,
        blurred_difference=blurred_difference,
        edge_map=edge_map,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
    )