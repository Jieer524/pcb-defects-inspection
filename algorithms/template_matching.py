import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from algorithms.common import preprocess_pair


@dataclass(frozen=True)
class TemplateMatchingDetection:
    """Outputs from one raw blockwise template matching comparison."""

    corr_threshold: float
    reference_gray: np.ndarray
    defective_gray: np.ndarray
    mask: np.ndarray
    boxes: list[dict[str, int | float]]
    processing_time_ms: float


def run_template_matching(
    reference_gray: np.ndarray,
    defective_gray: np.ndarray,
    threshold: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], float]:
    """Runs baseline template matching defect inspection between a reference template

    and a defective PCB image.

    Args:
        reference_gray: Preprocessed grayscale reference PCB template.
        defective_gray: Preprocessed grayscale defective PCB image.
        threshold: Threshold value to segment defect regions from the difference map.

    Returns:
        diff_map: Absolute difference map representing structural deviations.
        defect_mask: Binary mask of detected defects (255 for defect, 0 for background).
        bounding_boxes: List of bounding boxes as (x, y, w, h) for detected defects.
        processing_time_ms: Processing time in milliseconds.
    """
    if reference_gray is None or defective_gray is None:
        raise ValueError("Both reference_gray and defective_gray images are required.")

    start_time = time.perf_counter()

    # Calculate absolute structural difference between template and defective PCB
    diff_map = cv2.absdiff(reference_gray, defective_gray)

    # Apply thresholding to create a binary defect mask
    _, defect_mask = cv2.threshold(
        diff_map,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # Find contours for defect localisation and bounding box extraction
    contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bounding_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0

    return diff_map, defect_mask, bounding_boxes, processing_time_ms


def match_template_blockwise(
    reference_gray: np.ndarray,
    defective_gray: np.ndarray,
    block_size: tuple[int, int] = (64, 64),
    step_size: int = 32,
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], float]:
    """Performs localized block-based (patch-based) template matching using Normalized

    Cross-Correlation (NCC) to detect localized PCB defects.

    Args:
        reference_gray: Grayscale reference PCB template.
        defective_gray: Grayscale defective PCB image.
        block_size: Size of local template patches (height, width).
        step_size: Stride/step size for sliding the patch across the image.
        corr_threshold: Normalized cross-correlation score below which a block is marked defective.

    Returns:
        defect_mask: Binary mask highlighting blocks with low correlation.
        bounding_boxes: Bounding boxes (x, y, w, h) of defective blocks.
        processing_time_ms: Processing time in milliseconds.
    """
    if reference_gray is None or defective_gray is None:
        raise ValueError("Both reference_gray and defective_gray images are required.")

    start_time = time.perf_counter()

    height, width = reference_gray.shape[:2]
    block_h, block_w = block_size
    defect_mask = np.zeros((height, width), dtype=np.uint8)
    bounding_boxes: list[tuple[int, int, int, int]] = []

    for y in range(0, height - block_h + 1, step_size):
        for x in range(0, width - block_w + 1, step_size):
            ref_patch = reference_gray[y : y + block_h, x : x + block_w]
            def_patch = defective_gray[y : y + block_h, x : x + block_w]

            # Compute normalized correlation coefficient between patches
            result = cv2.matchTemplate(def_patch, ref_patch, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0]

            if score < corr_threshold:
                defect_mask[y : y + block_h, x : x + block_w] = 255
                bounding_boxes.append((x, y, block_w, block_h))

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0

    return defect_mask, bounding_boxes, processing_time_ms


def detect_template_matching(
    reference: np.ndarray,
    defective: np.ndarray,
    block_size: tuple[int, int] = (64, 64),
    step_size: int = 32,
    corr_threshold: float = 0.68,
) -> TemplateMatchingDetection:
    """Run the complete raw Blockwise Template Matching pipeline on an aligned image pair.

    Timing includes grayscale conversion, sliding window NCC computation, mask generation,
    and box construction. Disk I/O is excluded.
    """
    start_time = time.perf_counter()

    reference_gray, defective_gray = preprocess_pair(reference, defective)

    height, width = reference_gray.shape[:2]
    block_h, block_w = block_size
    defect_mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[dict[str, int | float]] = []

    for y in range(0, height - block_h + 1, step_size):
        for x in range(0, width - block_w + 1, step_size):
            ref_patch = reference_gray[y : y + block_h, x : x + block_w]
            def_patch = defective_gray[y : y + block_h, x : x + block_w]

            result = cv2.matchTemplate(def_patch, ref_patch, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])

            if score < corr_threshold:
                defect_mask[y : y + block_h, x : x + block_w] = 255
                boxes.append(
                    {
                        "xmin": int(x),
                        "ymin": int(y),
                        "xmax": int(x + block_w),
                        "ymax": int(y + block_h),
                        "contour_area": float(block_w * block_h),
                    }
                )

    processing_time_ms = (time.perf_counter() - start_time) * 1000.0

    return TemplateMatchingDetection(
        corr_threshold=corr_threshold,
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        mask=defect_mask,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
    )


def draw_defect_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draws defect bounding boxes onto an image for visualization.

    Args:
        image: Original BGR or grayscale PCB image.
        boxes: List of bounding boxes as (x, y, w, h).
        color: Box color in BGR format (default is Red).
        thickness: Line thickness for bounding boxes.

    Returns:
        annotated_image: Image with drawn defect bounding boxes.
    """
    if len(image.shape) == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()

    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

    return annotated
