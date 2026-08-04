import cv2
import numpy as np
import pytest

from algorithms.common import preprocess_pair, to_grayscale, validate_pair
from algorithms.evaluation import box_iou, evaluate_boxes
from algorithms.otsu import detect_otsu


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
