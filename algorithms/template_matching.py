"""Raw blockwise normalized-correlation template matching."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from algorithms.common import preprocess_pair


@dataclass(frozen=True)
class TemplateMatchingDetection:
    """Outputs from one raw blockwise template-matching comparison."""

    corr_threshold: float
    block_size: tuple[int, int]
    step_size: int
    reference_gray: np.ndarray
    defective_gray: np.ndarray
    similarity_map: np.ndarray
    mask: np.ndarray
    boxes: list[dict[str, int | float]]
    processing_time_ms: float


def _window_positions(length: int, block_length: int, step_size: int) -> list[int]:
    """Return stable window starts and include the final image border."""
    final_start = length - block_length
    positions = list(range(0, final_start + 1, step_size))
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def boxes_from_similarity_map(
    similarity_map: np.ndarray,
    image_shape: tuple[int, int],
    block_size: tuple[int, int],
    step_size: int,
    corr_threshold: float,
) -> list[dict[str, int | float]]:
    """Directly convert low-similarity grid cells to unsuppressed block boxes."""
    height, width = image_shape
    block_height, block_width = block_size
    y_positions = _window_positions(height, block_height, step_size)
    x_positions = _window_positions(width, block_width, step_size)
    if similarity_map.shape != (len(y_positions), len(x_positions)):
        raise ValueError("similarity_map shape does not match the window grid.")

    boxes: list[dict[str, int | float]] = []
    for row_index, column_index in np.argwhere(similarity_map < corr_threshold):
        y = y_positions[int(row_index)]
        x = x_positions[int(column_index)]
        boxes.append(
            {
                "xmin": int(x),
                "ymin": int(y),
                "xmax": int(x + block_width),
                "ymax": int(y + block_height),
                "similarity": float(similarity_map[row_index, column_index]),
            }
        )
    return boxes


def detect_template_matching(
    reference: np.ndarray,
    defective: np.ndarray,
    block_size: tuple[int, int] = (64, 64),
    step_size: int = 32,
    corr_threshold: float = 0.68,
) -> TemplateMatchingDetection:
    """Compare aligned patches using raw ``TM_CCOEFF_NORMED`` scores.

    Every low-similarity block becomes one prediction directly. Overlapping blocks
    are not merged, suppressed, filtered, or morphologically refined.
    """
    if not 0.0 <= corr_threshold <= 1.0:
        raise ValueError("corr_threshold must be between 0 and 1.")
    if len(block_size) != 2 or any(size <= 0 for size in block_size):
        raise ValueError("block_size must contain two positive integers.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    start_time = perf_counter()
    reference_gray, defective_gray = preprocess_pair(reference, defective)
    height, width = reference_gray.shape
    block_height, block_width = block_size
    if block_height > height or block_width > width:
        raise ValueError(
            f"block_size {block_size} exceeds image size {(height, width)}."
        )

    y_positions = _window_positions(height, block_height, step_size)
    x_positions = _window_positions(width, block_width, step_size)
    similarity_map = np.empty((len(y_positions), len(x_positions)), dtype=np.float32)
    for row_index, y in enumerate(y_positions):
        for column_index, x in enumerate(x_positions):
            reference_patch = reference_gray[
                y : y + block_height, x : x + block_width
            ]
            defective_patch = defective_gray[
                y : y + block_height, x : x + block_width
            ]
            score = float(
                cv2.matchTemplate(
                    defective_patch,
                    reference_patch,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            similarity_map[row_index, column_index] = score

    boxes = boxes_from_similarity_map(
        similarity_map,
        (height, width),
        block_size,
        step_size,
        corr_threshold,
    )
    defect_mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        defect_mask[
            int(box["ymin"]) : int(box["ymax"]),
            int(box["xmin"]) : int(box["xmax"]),
        ] = 255

    processing_time_ms = (perf_counter() - start_time) * 1000.0
    return TemplateMatchingDetection(
        corr_threshold=float(corr_threshold),
        block_size=(int(block_height), int(block_width)),
        step_size=int(step_size),
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        similarity_map=similarity_map,
        mask=defect_mask,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
    )
