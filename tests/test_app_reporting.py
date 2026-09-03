from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import (
    annotation_key,
    batch_summary_row,
    boxes_as_json,
    generate_batch_pdf,
    generate_pdf_certificate,
    optional_metrics,
    parse_uploaded_voc_boxes,
    run_detection,
)


class Upload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_uploaded_voc_annotation_and_metrics() -> None:
    upload = Upload(
        "board.xml",
        b"<annotation><filename>board.png</filename><object><name>defect</name>"
        b"<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>11</xmax><ymax>12</ymax>"
        b"</bndbox></object></annotation>",
    )
    truth = parse_uploaded_voc_boxes(upload)
    assert annotation_key(upload) == "board"
    assert truth[0]["class"] == "defect"
    metrics = optional_metrics([{"xmin": 1, "ymin": 2, "xmax": 11, "ymax": 12}], truth)
    assert metrics["precision"] == metrics["recall"] == metrics["f1_score"] == 1.0

    detection = SimpleNamespace(boxes=truth)
    payload = json.loads(
        boxes_as_json(
            "otsu",
            detection,
            {},
            ground_truth_boxes=truth,
            metrics=metrics,
            annotation_name="board.xml",
        )
    )
    assert payload["ground_truth_annotation"] == "board.xml"
    assert payload["evaluation_metrics"]["f1_score"] == 1.0


def test_metrics_are_unavailable_without_ground_truth() -> None:
    metrics = optional_metrics([], None)
    assert metrics["precision"] is None
    assert metrics["f1_score"] is None


def test_orb_web_dispatch_defaults_to_uncalibrated() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    config = {
        "orb": {
            "n_features": 5000,
            "scale_factor": 1.2,
            "n_levels": 8,
            "matcher_type": "bf_crosscheck",
            "spatial_distance_threshold": 15.0,
            "hamming_threshold": 40.0,
            "box_radius": 20,
            "merge_points": True,
            "morph_dilate": 0,
            "min_area": 300.0,
        }
    }
    with patch("app.detect_orb", return_value=SimpleNamespace()) as detector:
        run_detection("orb", image, image, config, {})
    assert detector.call_args.kwargs["calibrate"] is False


def test_single_and_batch_pdf_generation() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    detection = SimpleNamespace(
        boxes=[{"xmin": 1, "ymin": 2, "xmax": 11, "ymax": 12}],
        processing_time_ms=12.3,
    )
    metrics = optional_metrics(detection.boxes, detection.boxes)
    summary = pd.DataFrame(
        [batch_summary_row("board.png", detection, metrics, "Upload images")]
    )
    properties = pd.DataFrame([{"Defect #": 1, "Area (px^2)": 100}])
    preprocessing = {"enabled": True}
    algorithm_config = {"blur_ksize": 3}
    single = generate_pdf_certificate(
        "otsu",
        detection,
        image,
        image,
        image,
        preprocessing,
        algorithm_config,
        "reference.png",
        "board.png",
        properties,
        None,
    )
    batch = generate_batch_pdf(
        "otsu",
        "Upload images",
        summary,
        [{
            "image_name": "board.png",
            "summary": summary.iloc[0].to_dict(),
            "reference_rgb": image,
            "defective_rgb": image,
            "overlay_rgb": image,
            "metrics": metrics,
            "error": "",
        }],
        preprocessing,
        algorithm_config,
    )
    assert single.startswith(b"%PDF")
    assert batch.startswith(b"%PDF")
