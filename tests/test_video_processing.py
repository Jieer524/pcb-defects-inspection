"""Tests for video stream processing and conveyor simulation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import load_frozen_config, resolve_preprocessing_config, run_detection
from scripts.generate_conveyor_video import generate_conveyor_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "PCB_USED" / "01.JPG"
DEFECT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images" / "Missing_hole" / "01_missing_hole_01.jpg"
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """Generate a lightweight test conveyor video."""
    temp_dir = tmp_path_factory.mktemp("video_test")
    output_video = temp_dir / "test_conveyor.mp4"

    generate_conveyor_video(
        template_path=TEMPLATE_PATH,
        defect_paths=[DEFECT_PATH],
        output_path=output_video,
        target_width=640,
        target_height=334,
        fps=10,
        frames_per_board_inspection=5,
        frames_transition=3,
    )
    return output_video


def test_conveyor_video_creation(sample_video: Path):
    """Verify conveyor video generation produces a valid, readable video file."""
    assert sample_video.exists()
    assert sample_video.stat().st_size > 1000

    cap = cv2.VideoCapture(str(sample_video))
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert width == 640
        assert height == 334
        assert frame_count > 10
    finally:
        cap.release()


def test_frame_by_frame_inspection(sample_video: Path):
    """Verify that video frames can be ingested and inspected by the detection pipeline."""
    config = load_frozen_config()
    preprocessing_config = resolve_preprocessing_config(config)

    template = cv2.imread(str(TEMPLATE_PATH))
    assert template is not None

    cap = cv2.VideoCapture(str(sample_video))
    inspected_frames = 0
    statuses = []

    try:
        assert cap.isOpened()
        while cap.isOpened() and inspected_frames < 20:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize template to match frame if needed
            if template.shape[:2] != frame.shape[:2]:
                template_scaled = cv2.resize(
                    template, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA
                )
            else:
                template_scaled = template

            detection = run_detection(
                "otsu",
                template_scaled,
                frame,
                config,
                preprocessing_config,
            )

            defect_count = len(detection.boxes)
            statuses.append("DEFECT DETECTED" if defect_count > 0 else "PASSED")
            inspected_frames += 1
    finally:
        cap.release()

    assert inspected_frames > 5
    assert "DEFECT DETECTED" in statuses or "PASSED" in statuses


def test_video_detection_overlay():
    """Verify that HUD annotation and overlay functions handle video frame dimensions."""
    from app import draw_detection_overlay

    frame = np.zeros((335, 640, 3), dtype=np.uint8)
    boxes = [{"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "contour_area": 1600.0}]

    overlay = draw_detection_overlay(frame, boxes)
    assert overlay.shape == (335, 640, 3)

    # Verify red box was drawn (BGR: 0, 0, 255)
    # The bounding box lines should have red pixels
    assert np.any(overlay[:, :, 2] == 255)


def test_video_inspection_pdf_generation():
    """Verify that video stream inspection PDF report generates valid bytes."""
    import pandas as pd
    from app import FPDF, generate_video_inspection_pdf

    if FPDF is None:
        pytest.skip("fpdf2 not installed")

    board_df = pd.DataFrame([
        {
            "Board #": "Board #1",
            "Target Defect Category": "Clean Board (Normal)",
            "Time Window": "0.1s - 2.4s",
            "Inspection Station Defects": 0,
            "Quality Verdict": "PASS",
        },
        {
            "Board #": "Board #2",
            "Target Defect Category": "Missing Hole",
            "Time Window": "2.5s - 4.9s",
            "Inspection Station Defects": 3,
            "Quality Verdict": "REJECT",
        },
    ])

    frame_df = pd.DataFrame([
        {
            "Board #": "Board #1",
            "Defect Category": "Clean Board (Normal)",
            "Frame": 2,
            "Timestamp (s)": 0.1,
            "Status": "PASSED",
            "Defects Detected": 0,
            "Inference Time (ms)": 2.5,
        },
        {
            "Board #": "Board #2",
            "Defect Category": "Missing Hole",
            "Frame": 84,
            "Timestamp (s)": 4.2,
            "Status": "DEFECT DETECTED",
            "Defects Detected": 3,
            "Inference Time (ms)": 2.3,
        },
    ])

    sample_img = np.zeros((300, 600, 3), dtype=np.uint8)

    pdf_bytes = generate_video_inspection_pdf(
        algorithm="otsu",
        video_name="conveyor_test.mp4",
        template_name="01.JPG",
        board_summary_df=board_df,
        frame_log_df=frame_df,
        sample_annotated_frame=sample_img,
        reference_rgb=sample_img,
        preprocessing_config={"denoise": "median"},
        algorithm_config={"min_area": 150},
        avg_fps=400.0,
        inspected_count=100,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
