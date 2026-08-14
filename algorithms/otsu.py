"""Raw Otsu defect detection without denoising or post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from algorithms.common import preprocess_pair
from algorithms.contours import extract_external_boxes


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
    boxes = extract_external_boxes(mask)

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
