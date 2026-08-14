"""Raw ORB feature detection and matching for aligned PCB defect inspection."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from algorithms.common import preprocess_pair


@dataclass(frozen=True)
class ORBDetection:
    """Outputs and diagnostic metrics from one raw ORB comparison."""

    reference_gray: np.ndarray
    defective_gray: np.ndarray
    num_reference_keypoints: int
    num_defective_keypoints: int
    num_matches: int
    num_consistent_matches: int
    num_inconsistent_matches: int
    num_unmatched_keypoints: int
    defect_points: list[tuple[float, float]]
    boxes: list[dict[str, int | float]]
    processing_time_ms: float
    n_features: int
    spatial_distance_threshold: float
    hamming_threshold: float
    box_radius: int
    matcher_type: str


def extract_raw_boxes_from_points(
    points: Sequence[tuple[float, float]],
    image_shape: tuple[int, int],
    box_radius: int = 25,
) -> list[dict[str, int | float]]:
    """Convert defect coordinate points to raw bounding boxes.

    Each detected anomaly coordinate (x, y) forms a fixed-radius bounding box
    [x - r, y - r, x + r, y + r] clamped to the image boundary. In adherence to
    the project's raw evaluation protocol, no clustering, morphological dilation,
    or merging is applied.
    """
    if box_radius <= 0:
        raise ValueError("box_radius must be a positive integer.")

    height, width = image_shape[:2]
    boxes: list[dict[str, int | float]] = []

    for x, y in points:
        xmin = max(0, int(round(x - box_radius)))
        ymin = max(0, int(round(y - box_radius)))
        xmax = min(width, int(round(x + box_radius)))
        ymax = min(height, int(round(y + box_radius)))

        if xmax > xmin and ymax > ymin:
            boxes.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "center_x": float(x),
                    "center_y": float(y),
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


def detect_orb(
    reference: np.ndarray,
    defective: np.ndarray,
    n_features: int = 5000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
    spatial_distance_threshold: float = 15.0,
    hamming_threshold: float = 60.0,
    box_radius: int = 35,
    matcher_type: str = "bf_crosscheck",
    ratio_threshold: float = 0.75,
) -> ORBDetection:
    """Run the raw ORB feature detection and matching pipeline on an aligned image pair.

    Workflow:
    1. Neutral grayscale conversion via `preprocess_pair`.
    2. Extract ORB keypoints and 256-bit binary descriptors from both images.
    3. Match descriptors using Hamming distance (Brute-Force with cross-check or KNN ratio test).
    4. Identify defect points:
       - Matches with spatial displacement > `spatial_distance_threshold`
       - Matches with descriptor Hamming distance > `hamming_threshold`
       - Unmatched keypoints on the defective image
    5. Directly form raw predicted bounding boxes around defect points without filtering/merging.
    """
    if matcher_type not in ("bf_crosscheck", "knn_ratio"):
        raise ValueError(
            f"Unsupported matcher_type '{matcher_type}'. Use 'bf_crosscheck' or 'knn_ratio'."
        )
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1.")
    if n_levels <= 0:
        raise ValueError("n_levels must be positive.")
    if spatial_distance_threshold < 0 or hamming_threshold < 0:
        raise ValueError("Spatial and Hamming thresholds must be non-negative.")
    if box_radius <= 0:
        raise ValueError("box_radius must be positive.")
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1.")

    start_time = perf_counter()

    reference_gray, defective_gray = preprocess_pair(reference, defective)
    image_shape = defective_gray.shape[:2]

    # Initialize raw OpenCV ORB detector
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
    )

    kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
    kp_def, des_def = orb.detectAndCompute(defective_gray, None)

    num_ref_kp = len(kp_ref) if kp_ref is not None else 0
    num_def_kp = len(kp_def) if kp_def is not None else 0

    defect_points: list[tuple[float, float]] = []
    num_matches = 0
    num_consistent = 0
    num_inconsistent = 0
    num_unmatched = 0

    has_descriptors = (
        des_ref is not None
        and des_def is not None
        and len(des_ref) > 0
        and len(des_def) > 0
    )

    if has_descriptors:
        if matcher_type == "bf_crosscheck":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_ref, des_def)
            num_matches = len(matches)

            matched_def_indices: set[int] = set()
            for m in matches:
                matched_def_indices.add(m.trainIdx)
                pt_ref = np.asarray(kp_ref[m.queryIdx].pt, dtype=np.float64)
                pt_def = np.asarray(kp_def[m.trainIdx].pt, dtype=np.float64)
                disp = float(np.linalg.norm(pt_def - pt_ref))

                if disp > spatial_distance_threshold or m.distance > hamming_threshold:
                    num_inconsistent += 1
                    defect_points.append((float(pt_def[0]), float(pt_def[1])))
                else:
                    num_consistent += 1

            # Unmatched defective keypoints
            for idx, kp in enumerate(kp_def):
                if idx not in matched_def_indices:
                    num_unmatched += 1
                    defect_points.append((float(kp.pt[0]), float(kp.pt[1])))

        elif matcher_type == "knn_ratio":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn_matches = bf.knnMatch(des_def, des_ref, k=2)
            num_matches = len(knn_matches)

            matched_def_indices: set[int] = set()
            for pair in knn_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio_threshold * n.distance:
                        matched_def_indices.add(m.queryIdx)
                        pt_def = np.asarray(kp_def[m.queryIdx].pt, dtype=np.float64)
                        pt_ref = np.asarray(kp_ref[m.trainIdx].pt, dtype=np.float64)
                        disp = float(np.linalg.norm(pt_def - pt_ref))

                        if (
                            disp > spatial_distance_threshold
                            or m.distance > hamming_threshold
                        ):
                            num_inconsistent += 1
                            defect_points.append((float(pt_def[0]), float(pt_def[1])))
                        else:
                            num_consistent += 1

            for idx, kp in enumerate(kp_def):
                if idx not in matched_def_indices:
                    num_unmatched += 1
                    defect_points.append((float(kp.pt[0]), float(kp.pt[1])))
    else:
        # If one image has keypoints but no descriptors could be matched
        if kp_def is not None:
            num_unmatched = len(kp_def)
            for kp in kp_def:
                defect_points.append((float(kp.pt[0]), float(kp.pt[1])))

    boxes = extract_raw_boxes_from_points(
        defect_points,
        image_shape=image_shape,
        box_radius=box_radius,
    )

    processing_time_ms = (perf_counter() - start_time) * 1000.0

    return ORBDetection(
        reference_gray=reference_gray,
        defective_gray=defective_gray,
        num_reference_keypoints=num_ref_kp,
        num_defective_keypoints=num_def_kp,
        num_matches=num_matches,
        num_consistent_matches=num_consistent,
        num_inconsistent_matches=num_inconsistent,
        num_unmatched_keypoints=num_unmatched,
        defect_points=defect_points,
        boxes=boxes,
        processing_time_ms=processing_time_ms,
        n_features=n_features,
        spatial_distance_threshold=spatial_distance_threshold,
        hamming_threshold=hamming_threshold,
        box_radius=box_radius,
        matcher_type=matcher_type,
    )
