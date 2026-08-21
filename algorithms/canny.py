"""Raw Canny defect detection without denoising or post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from algorithms.common import preprocess_pair
from algorithms.contours import extract_external_boxes


@dataclass(frozen=True)
class CannyDetection:
    """Outputs from one raw Canny comparison."""

    low_threshold: float
    high_threshold: float
    aperture_size: int
    l2_gradient: bool
    reference_gray: np.ndarray
    defective_gray: np.ndarray
    reference_edges: np.ndarray
    defective_edges: np.ndarray
    edge_difference: np.ndarray
    boxes: list[dict[str, int | float]]
    processing_time_ms: float


def detect_canny(
    reference: np.ndarray,
    defective: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = False,
    preprocessing_config: dict | None = None,
    morph_dilate: int = 0,
    morph_close: int = 0,
    min_area: float = 0.0,
) -> CannyDetection:
    """Apply Canny independently, then compare the two edge maps.

    An optional ``preprocessing_config`` dict is passed to ``preprocess_pair`` so
    the two grayscale images can be denoised and contrast-enhanced before edge
    detection. ``None`` preserves the original raw behaviour.

    Enhancement parameters:
        morph_dilate: Kernel size for morphological dilation to expand thin edges
            into solid regions matching annotation boundaries.
        morph_close: Kernel size for morphological closing to bridge narrow gaps.
        min_area: Minimum contour area threshold to filter isolated edge noise.
    """
    if low_threshold < 0 or high_threshold <= low_threshold:
        raise ValueError("Canny thresholds require 0 <= low < high.")
    if aperture_size not in (3, 5, 7):
        raise ValueError("Canny aperture_size must be 3, 5, or 7.")

    start_time = perf_counter()
    reference_gray, defective_gray = preprocess_pair(
        reference, defective, preprocessing_config
    )
    reference_edges = cv2.Canny(
        reference_gray,
        low_threshold,
        high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    defective_edges = cv2.Canny(
        defective_gray,
        low_threshold,
        high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    edge_difference = cv2.absdiff(reference_edges, defective_edges)

    # --- Enhancement: morphological post-processing ---
    mask = edge_difference.copy()
    if morph_close > 0:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_close, morph_close)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    if morph_dilate > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_dilate, morph_dilate)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate)

    # --- Enhancement: contour extraction with area filtering ---
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

    return CannyDetection(
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
        aperture_size=aperture_size,
        l2_gradient=l2_gradient,
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        reference_edges=reference_edges,
        defective_edges=defective_edges,
        edge_difference=edge_difference,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
    )

