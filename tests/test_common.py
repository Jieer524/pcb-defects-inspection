import cv2
import numpy as np
import pytest

from algorithms.canny import detect_canny
from algorithms.common import preprocess_pair, to_grayscale, validate_pair
from algorithms.evaluation import box_iou, evaluate_boxes
from algorithms.otsu import detect_otsu
from algorithms.orb import extract_raw_boxes_from_points
from algorithms.template_matching import detect_template_matching


def test_preprocess_pair_converts_without_blurring() -> None:
    reference = np.zeros((5, 5, 3), dtype=np.uint8)
    defective = reference.copy()
    defective[2, 2] = (255, 255, 255)

    reference_gray, defective_gray = preprocess_pair(reference, defective)

    assert reference_gray.shape == (5, 5)
    assert defective_gray[2, 2] == 255
    assert cv2.countNonZero(defective_gray) == 1


def test_validate_pair_rejects_dimension_mismatch() -> None:
    reference = np.zeros((10, 10, 3), dtype=np.uint8)
    defective = np.zeros((8, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="dimensions must match"):
        validate_pair(reference, defective)


def test_to_grayscale_preserves_grayscale_values() -> None:
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)

    result = to_grayscale(image)

    np.testing.assert_array_equal(result, image)
    assert result is not image


def test_raw_otsu_detects_an_unfiltered_region() -> None:
    reference = np.zeros((20, 20, 3), dtype=np.uint8)
    defective = reference.copy()
    defective[5:10, 7:13] = 255

    result = detect_otsu(reference, defective)

    assert result.threshold == 0.0
    assert result.boxes == [
        {
            "xmin": 7,
            "ymin": 5,
            "xmax": 13,
            "ymax": 10,
            "contour_area": 20.0,
        }
    ]
    assert cv2.countNonZero(result.mask) == 30


def test_box_metrics_use_one_to_one_matching() -> None:
    ground_truth = [{"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}]
    predictions = [
        {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
        {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
    ]

    assert box_iou(predictions[0], ground_truth[0]) == 1.0
    metrics = evaluate_boxes(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0


def test_raw_canny_compares_independent_edge_maps() -> None:
    reference = np.zeros((30, 30, 3), dtype=np.uint8)
    defective = reference.copy()
    defective[8:20, 10:22] = 255

    result = detect_canny(reference, defective, 50, 150)

    assert cv2.countNonZero(result.reference_edges) == 0
    assert cv2.countNonZero(result.defective_edges) > 0
    np.testing.assert_array_equal(result.edge_difference, result.defective_edges)
    assert result.boxes


def test_raw_canny_rejects_invalid_threshold_order() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="low < high"):
        detect_canny(image, image, 150, 50)


def test_template_matching_returns_direct_low_similarity_blocks() -> None:
    reference = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    defective = reference.copy()
    defective[:8, :8] = cv2.flip(defective[:8, :8], 1)

    result = detect_template_matching(
        reference,
        defective,
        block_size=(8, 8),
        step_size=8,
        corr_threshold=0.9,
    )

    assert result.similarity_map.shape == (2, 2)
    assert result.boxes
    assert all(float(box["similarity"]) < 0.9 for box in result.boxes)


def test_template_matching_rejects_invalid_block_size() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="exceeds image size"):
        detect_template_matching(image, image, block_size=(20, 20))


def test_orb_point_boxes_are_direct_and_unmerged() -> None:
    boxes = extract_raw_boxes_from_points(
        [(10.0, 10.0), (12.0, 10.0)],
        image_shape=(30, 30),
        box_radius=5,
    )

    assert len(boxes) == 2
    assert boxes[0]["xmin"] == 5
    assert boxes[1]["xmin"] == 7
