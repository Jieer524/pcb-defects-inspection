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


def detect_otsu(
    reference: np.ndarray,
    defective: np.ndarray,
    preprocessing_config: dict | None = None,
    blur_ksize: int = 0,
    morph_open: int = 0,
    morph_dilate: int = 0,
    min_area: float = 0.0,
) -> OtsuDetection:
    """Run the Otsu pipeline with optional preprocessing and morphology enhancements."""
    start_time = perf_counter()

    reference_gray, defective_gray = preprocess_pair(
        reference, defective, preprocessing_config
    )
    difference = cv2.absdiff(reference_gray, defective_gray)
    diff_proc = difference
    if blur_ksize > 1:
        diff_proc = cv2.medianBlur(difference, blur_ksize)

    threshold, mask = cv2.threshold(
        diff_proc,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if morph_open > 0:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_open, morph_open)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    if morph_dilate > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_dilate, morph_dilate)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
    boxes.sort(
        key=lambda box: (
            int(box["ymin"]),
            int(box["xmin"]),
            int(box["ymax"]),
            int(box["xmax"]),
        )
    )

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
