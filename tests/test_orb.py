import numpy as np
import pytest

from algorithms.orb import detect_orb, extract_raw_boxes_from_points


def test_extract_raw_boxes_from_points() -> None:
    points = [(10.0, 20.0), (0.0, 0.0)]
    boxes = extract_raw_boxes_from_points(points, image_shape=(100, 100), box_radius=5)

    assert len(boxes) == 2
    # Sorted by ymin, xmin
    assert boxes[0]["xmin"] == 0
    assert boxes[0]["ymin"] == 0
    assert boxes[0]["xmax"] == 5
    assert boxes[0]["ymax"] == 5

    assert boxes[1]["xmin"] == 5
    assert boxes[1]["ymin"] == 15
    assert boxes[1]["xmax"] == 15
    assert boxes[1]["ymax"] == 25


def test_extract_raw_boxes_invalid_radius() -> None:
    with pytest.raises(ValueError, match="box_radius must be a positive integer"):
        extract_raw_boxes_from_points([(10.0, 10.0)], (100, 100), box_radius=0)


def test_detect_orb_identical_images() -> None:
    # Synthesize an image with some corners
    np.random.seed(42)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(10, 190, 20):
        img[i : i + 10, i : i + 10] = 255

    result = detect_orb(img, img.copy())

    assert result.num_reference_keypoints > 0
    assert result.num_defective_keypoints > 0
    # For identical images, consistent matches should be high and inconsistent matches should be 0 or very small
    assert result.num_consistent_matches > 0
    assert result.processing_time_ms > 0


def test_detect_orb_invalid_matcher() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported matcher_type"):
        detect_orb(img, img, matcher_type="invalid")
