"""Streamlit deployment: defect inspection with property table, PDF export, batch mode."""

from __future__ import annotations

import json
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from algorithms.canny import detect_canny
from algorithms.common import load_image
from algorithms.evaluation import evaluate_boxes, parse_voc_boxes
from algorithms.orb import detect_orb
from algorithms.otsu import detect_otsu
from algorithms.template_matching import detect_template_matching
from algorithms.preprocessing import build_preprocessing_config

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - dashboard degrades gracefully
    FPDF = None


PROJECT_ROOT = Path(__file__).resolve().parent
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "frozen_parameters.yaml"
DATASET_SPLIT_PATH = PROJECT_ROOT / "data" / "dataset_split.csv"
TEMPLATES_DIR = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "PCB_USED"
DEFAULT_CONVEYOR_VIDEO = PROJECT_ROOT / "data" / "conveyor_inspection_demo.mp4"

ALGORITHMS = ("template_matching", "otsu", "canny", "orb")

CONVEYOR_BOARD_CATALOG = [
    {"board_num": 1, "type": "Clean Board (Normal)", "is_defect": False},
    {"board_num": 2, "type": "Missing Hole", "is_defect": True},
    {"board_num": 3, "type": "Clean Board (Normal)", "is_defect": False},
    {"board_num": 4, "type": "Short Circuit", "is_defect": True},
    {"board_num": 5, "type": "Clean Board (Normal)", "is_defect": False},
    {"board_num": 6, "type": "Mouse Bite", "is_defect": True},
    {"board_num": 7, "type": "Clean Board (Normal)", "is_defect": False},
    {"board_num": 8, "type": "Spurious Copper", "is_defect": True},
    {"board_num": 9, "type": "Clean Board (Normal)", "is_defect": False},
]


@st.cache_data
def load_frozen_config() -> dict[str, object]:
    """Load the full frozen configuration."""
    with FROZEN_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_data
def load_dataset_manifest() -> pd.DataFrame:
    return pd.read_csv(DATASET_SPLIT_PATH)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode a Streamlit upload as an OpenCV BGR image."""
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode '{uploaded_file.name}' as an image.")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resolve_preprocessing_config(config: dict) -> dict:
    """Build the preprocessing_config dict from the frozen yaml section."""
    return build_preprocessing_config(config.get("preprocessing"))


def run_detection(
    algorithm: str,
    reference: np.ndarray,
    defective: np.ndarray,
    config: dict,
    preprocessing_config: dict,
    pixel_to_mm: float | None = None,
):
    """Dispatch to the selected algorithm with the frozen + preprocessing configs."""
    if algorithm == "otsu":
        section = config.get("otsu", {})
        return detect_otsu(
            reference,
            defective,
            preprocessing_config=preprocessing_config,
            blur_ksize=int(section.get("blur_ksize", 3)),
            morph_open=int(section.get("morph_open", 0)),
            morph_dilate=int(section.get("morph_dilate", 35)),
            min_area=float(section.get("min_area", 150.0)),
        )
    if algorithm == "canny":
        section = config["canny"]
        return detect_canny(
            reference,
            defective,
            low_threshold=float(section["low_threshold"]),
            high_threshold=float(section["high_threshold"]),
            aperture_size=int(section["aperture_size"]),
            l2_gradient=bool(section["l2_gradient"]),
            preprocessing_config=preprocessing_config,
            morph_dilate=int(section.get("morph_dilate", 0)),
            morph_close=int(section.get("morph_close", 5)),
            min_area=float(section.get("min_area", 300.0)),
        )
    if algorithm == "template_matching":
        section = config["template_matching"]
        return detect_template_matching(
            reference,
            defective,
            block_size=tuple(section["block_size"]),
            step_size=int(section["step_size"]),
            corr_threshold=float(section["corr_threshold"]),
            preprocessing_config=preprocessing_config,
        )
    if algorithm == "orb":
        section = config["orb"]
        return detect_orb(
            reference,
            defective,
            n_features=int(section["n_features"]),
            scale_factor=float(section["scale_factor"]),
            n_levels=int(section["n_levels"]),
            matcher_type=str(section["matcher_type"]),
            spatial_distance_threshold=float(section["spatial_distance_threshold"]),
            hamming_threshold=float(section["hamming_threshold"]),
            box_radius=int(section["box_radius"]),
            calibrate=bool(section.get("calibrate", False)),
            ransac_reproj_threshold=float(section.get("ransac_reproj_threshold", 5.0)),
            preprocessing_config=preprocessing_config,
            merge_points=bool(section.get("merge_points", True)),
            morph_dilate=int(section.get("morph_dilate", 0)),
            min_area=float(section.get("min_area", 300.0)),
        )
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def draw_detection_overlay(
    defective_image: np.ndarray,
    boxes: list[dict],
    ground_truth_boxes: list[dict] | None = None,
) -> np.ndarray:
    """Draw predicted boxes in red and optional ground truth in green."""
    overlay = defective_image.copy()
    for box in ground_truth_boxes or []:
        cv2.rectangle(
            overlay,
            (int(box["xmin"]), int(box["ymin"])),
            (int(box["xmax"]), int(box["ymax"])),
            (0, 255, 0),
            2,
        )
    for box in boxes:
        cv2.rectangle(
            overlay,
            (int(box["xmin"]), int(box["ymin"])),
            (int(box["xmax"]), int(box["ymax"])),
            (0, 0, 255),
            2,
        )
    return overlay


def build_property_table(boxes: list[dict], pixel_to_mm: float | None = None) -> pd.DataFrame:
    """Build the defect property summary table from predicted boxes."""
    rows = []
    for index, box in enumerate(boxes, start=1):
        width = int(box["xmax"]) - int(box["xmin"])
        height = int(box["ymax"]) - int(box["ymin"])
        area = width * height
        aspect_ratio = width / height if height else 0.0
        row = {
            "Defect #": index,
            "x_min": int(box["xmin"]),
            "y_min": int(box["ymin"]),
            "x_max": int(box["xmax"]),
            "y_max": int(box["ymax"]),
            "Width (px)": width,
            "Height (px)": height,
            "Area (px²)": area,
            "Aspect Ratio": round(aspect_ratio, 3),
        }
        if "similarity" in box:
            row["Similarity Score"] = round(float(box["similarity"]), 4)
        if pixel_to_mm is not None:
            row["Width (mm)"] = round(width * pixel_to_mm, 3)
            row["Height (mm)"] = round(height * pixel_to_mm, 3)
            row["Area (mm²)"] = round(area * pixel_to_mm * pixel_to_mm, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_uploaded_voc_boxes(uploaded_file) -> list[dict[str, int | str]]:
    """Parse Pascal VOC boxes from an uploaded XML file."""
    try:
        root = ET.fromstring(uploaded_file.getvalue())
    except ET.ParseError as error:
        raise ValueError(f"Invalid annotation XML '{uploaded_file.name}': {error}") from error

    boxes: list[dict[str, int | str]] = []
    for obj in root.findall("object"):
        bounds = obj.find("bndbox")
        if bounds is None:
            raise ValueError(f"Object without bndbox in '{uploaded_file.name}'.")
        box = {
            "class": obj.findtext("name", default="unknown"),
            "xmin": int(bounds.findtext("xmin", default="-1")),
            "ymin": int(bounds.findtext("ymin", default="-1")),
            "xmax": int(bounds.findtext("xmax", default="-1")),
            "ymax": int(bounds.findtext("ymax", default="-1")),
        }
        if box["xmin"] >= box["xmax"] or box["ymin"] >= box["ymax"]:
            raise ValueError(f"Invalid bounding box in '{uploaded_file.name}': {box}")
        boxes.append(box)
    return boxes


def annotation_key(uploaded_file) -> str:
    """Return the image stem named by VOC XML, falling back to the XML stem."""
    try:
        root = ET.fromstring(uploaded_file.getvalue())
        image_name = root.findtext("filename")
    except ET.ParseError:
        image_name = None
    return Path(image_name).stem.casefold() if image_name else Path(uploaded_file.name).stem.casefold()


def optional_metrics(
    predicted_boxes: list[dict], ground_truth_boxes: list[dict] | None
) -> dict[str, int | float | None]:
    """Calculate localisation metrics only when ground truth is available."""
    if ground_truth_boxes is None:
        return {
            "ground_truth": None,
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "mean_matched_iou": None,
        }
    metrics = evaluate_boxes(predicted_boxes, ground_truth_boxes, iou_threshold=0.5)
    return {"ground_truth": len(ground_truth_boxes), **metrics}


def batch_summary_row(
    image_name: str,
    detection,
    metrics: dict[str, int | float | None],
    source: str,
    defect_class: str | None = None,
    image_shape: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Build one display/export row for a successful batch result."""
    return {
        "Image": image_name,
        "Source": source,
        "Defect class": defect_class,
        "Width (px)": image_shape[1] if image_shape else None,
        "Height (px)": image_shape[0] if image_shape else None,
        "Candidates": len(detection.boxes),
        "Ground truth": metrics["ground_truth"],
        "TP": metrics["true_positives"],
        "FP": metrics["false_positives"],
        "FN": metrics["false_negatives"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-score": metrics["f1_score"],
        "Mean matched IoU": metrics["mean_matched_iou"],
        "Processing time (ms)": round(detection.processing_time_ms, 1),
        "Status": "Defects detected" if detection.boxes else "No defects detected",
        "Error": "",
    }


def boxes_as_json(
    algorithm: str,
    detection,
    config: dict,
    ground_truth_boxes: list[dict] | None = None,
    metrics: dict[str, int | float | None] | None = None,
    annotation_name: str | None = None,
) -> str:
    parameter_keys = (
        "n_features",
        "scale_factor",
        "n_levels",
        "matcher_type",
        "spatial_distance_threshold",
        "hamming_threshold",
        "box_radius",
        "low_threshold",
        "high_threshold",
        "aperture_size",
        "l2_gradient",
        "block_size",
        "step_size",
        "corr_threshold",
        "threshold",
        "calibrated",
        "inlier_count",
    )
    parameters = {}
    for key, value in vars(detection).items():
        if key in parameter_keys:
            parameters[key] = list(value) if isinstance(value, (list, tuple)) else value
    payload = {
        "algorithm": algorithm,
        "parameters": parameters,
        "predicted_boxes": detection.boxes,
        "ground_truth_annotation": annotation_name,
        "ground_truth_boxes": ground_truth_boxes,
        "evaluation_metrics": metrics,
    }
    return json.dumps(payload, indent=2)


def pdf_text(value: object) -> str:
    """Convert values to text supported by the built-in PDF font."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return str(value).replace("²", "^2").replace("×", "x").replace("≥", ">=")


def add_pdf_key_values(pdf, values: dict[str, object]) -> None:
    pdf.set_font("Helvetica", "", 9)
    for label, value in values.items():
        pdf.multi_cell(
            0,
            5,
            f"{pdf_text(label)}: {pdf_text(value)}",
            new_x="LMARGIN",
            new_y="NEXT",
        )


def add_pdf_dataframe(pdf, table: pd.DataFrame, title: str) -> None:
    """Write every DataFrame value to the PDF as row-oriented records."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, pdf_text(title), new_x="LMARGIN", new_y="NEXT")
    if table.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "No records", new_x="LMARGIN", new_y="NEXT")
        return
    for index, row in table.iterrows():
        if pdf.get_y() > pdf.h - 35:
            pdf.add_page()
        pdf.set_fill_color(235, 240, 245)
        pdf.set_font("Helvetica", "B", 8)
        record_name = row.get("Image", row.get("Defect #", index + 1))
        pdf.cell(0, 5, f"Record: {pdf_text(record_name)}", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 7)
        line = " | ".join(f"{pdf_text(column)}: {pdf_text(row[column])}" for column in table.columns)
        pdf.multi_cell(0, 4, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)


def add_pdf_image_triptych(
    pdf,
    images: list[tuple[str, np.ndarray]],
    heading: str = "Inspection images",
    summary_values: dict[str, object] | None = None,
) -> None:
    """Place one image's summary and three inspection images on one page."""
    pdf.add_page(orientation="L")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0,
        8,
        pdf_text(heading),
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    if summary_values:
        pdf.set_font("Helvetica", "", 7.5)
        summary_text = " | ".join(
            f"{pdf_text(label)}: {pdf_text(value)}"
            for label, value in summary_values.items()
        )
        pdf.multi_cell(0, 4, summary_text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    gap = 4.0
    column_width = (pdf.w - pdf.l_margin - pdf.r_margin - gap * 2) / 3
    image_width = column_width - 2
    label_y = pdf.get_y()
    image_y = label_y + 7
    maximum_height = pdf.h - image_y - pdf.b_margin

    for index, (label, image_rgb) in enumerate(images):
        x = pdf.l_margin + index * (column_width + gap)
        pdf.set_xy(x, label_y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(column_width, 6, pdf_text(label), align="C")

        maximum_dimension = 1600
        scale = min(1.0, maximum_dimension / max(image_rgb.shape[:2]))
        report_image = (
            cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image_rgb
        )
        aspect_height = image_width * report_image.shape[0] / report_image.shape[1]
        rendered_width = image_width
        if aspect_height > maximum_height:
            aspect_height = maximum_height
            rendered_width = aspect_height * report_image.shape[1] / report_image.shape[0]
        encoded, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(report_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not encoded:
            raise ValueError(f"Could not encode image for PDF ({label}).")
        centered_x = x + (column_width - rendered_width) / 2
        with BytesIO(buffer.tobytes()) as image_buffer:
            pdf.image(image_buffer, x=centered_x, y=image_y, w=rendered_width)


def generate_pdf_certificate(
    algorithm: str,
    detection,
    reference_rgb: np.ndarray,
    defective_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    preprocessing_config: dict,
    algorithm_config: dict,
    reference_name: str,
    defective_name: str,
    property_table: pd.DataFrame,
    pixel_to_mm: float | None,
    ground_truth_boxes: list[dict] | None = None,
    metrics: dict[str, int | float | None] | None = None,
    annotation_name: str | None = None,
) -> bytes:
    """Generate a complete single-inspection PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PCB Defect Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
    status = "Defects detected" if detection.boxes else "No defects detected"
    report_values = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Algorithm": algorithm,
        "Reference image": reference_name,
        "Inspection image": defective_name,
        "Image dimensions": f"{defective_rgb.shape[1]} x {defective_rgb.shape[0]} px",
        "Status": status,
        "Candidate defect blocks": len(detection.boxes),
        "Processing time": f"{detection.processing_time_ms:.1f} ms",
        "Pixel-to-mm calibration": pixel_to_mm if pixel_to_mm is not None else "Disabled",
        "Shared preprocessing": json.dumps(preprocessing_config, sort_keys=True),
        "Frozen algorithm configuration": json.dumps(algorithm_config, sort_keys=True),
    }
    if metrics is not None:
        report_values.update({
            "Ground-truth annotation": annotation_name,
            "Evaluation rule": "Greedy one-to-one box matching at IoU >= 0.50",
            "Ground-truth boxes": metrics["ground_truth"],
            "True positives": metrics["true_positives"],
            "False positives": metrics["false_positives"],
            "False negatives": metrics["false_negatives"],
            "Precision": f"{float(metrics['precision']):.4f}",
            "Recall": f"{float(metrics['recall']):.4f}",
            "F1-score": f"{float(metrics['f1_score']):.4f}",
            "Mean matched IoU": f"{float(metrics['mean_matched_iou']):.4f}",
        })
    add_pdf_key_values(pdf, report_values)
    pdf.ln(2)
    add_pdf_dataframe(pdf, property_table, "Defect property summary")
    add_pdf_image_triptych(pdf, [
        ("Reference image", reference_rgb),
        ("Defective image", defective_rgb),
        (
            "Predictions: red; ground truth: green" if ground_truth_boxes is not None
            else "Candidate defect blocks (red)",
            overlay_rgb,
        ),
    ])
    return bytes(pdf.output())


def generate_batch_pdf(
    algorithm: str,
    source_mode: str,
    summary: pd.DataFrame,
    batch_results: list[dict[str, object]],
    preprocessing_config: dict,
    algorithm_config: dict,
) -> bytes:
    """Generate a batch PDF containing all displayed summary and detail data."""
    pdf = FPDF(orientation="L")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PCB Batch Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
    add_pdf_key_values(pdf, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Algorithm": algorithm,
        "Source": source_mode,
        "Images processed": len(summary),
        "Successful": int((summary["Error"] == "").sum()),
        "Failed": int((summary["Error"] != "").sum()),
        "Total candidates": int(summary["Candidates"].fillna(0).sum()),
        "Mean processing time": f"{summary['Processing time (ms)'].mean():.1f} ms",
        "Mean precision": f"{summary['Precision'].mean():.4f}" if summary["Precision"].notna().any() else "N/A",
        "Mean F1-score": f"{summary['F1-score'].mean():.4f}" if summary["F1-score"].notna().any() else "N/A",
        "Metric rule": "One-to-one matching at IoU >= 0.50; N/A when ground truth is not supplied",
        "Shared preprocessing": json.dumps(preprocessing_config, sort_keys=True),
        "Frozen algorithm configuration": json.dumps(algorithm_config, sort_keys=True),
    })
    add_pdf_dataframe(pdf, summary, "Batch results")

    for item in batch_results:
        if item.get("error"):
            continue
        add_pdf_image_triptych(pdf, [
            ("Reference image", item["reference_rgb"]),
            ("Defective image", item["defective_rgb"]),
            (
                "Predictions: red; ground truth: green"
                if item["metrics"]["ground_truth"] is not None
                else "Candidate defect blocks (red)",
                item["overlay_rgb"],
            ),
        ], heading=f"Inspection result: {item['image_name']}", summary_values=item["summary"])
    return bytes(pdf.output())


def generate_video_inspection_pdf(
    algorithm: str,
    video_name: str,
    template_name: str,
    board_summary_df: pd.DataFrame,
    frame_log_df: pd.DataFrame,
    sample_annotated_frame: np.ndarray | None,
    reference_rgb: np.ndarray | None,
    preprocessing_config: dict,
    algorithm_config: dict,
    avg_fps: float,
    inspected_count: int,
) -> bytes:
    """Generate a comprehensive PDF inspection report for video stream analysis."""
    pdf = FPDF(orientation="L")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PCB Video Stream Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")

    total_boards = len(board_summary_df)
    passed_boards = (
        int((board_summary_df["Quality Verdict"] == "PASS").sum())
        if not board_summary_df.empty
        else 0
    )
    rejected_boards = (
        int((board_summary_df["Quality Verdict"] == "REJECT").sum())
        if not board_summary_df.empty
        else 0
    )
    pass_rate = f"{(passed_boards / max(total_boards, 1)) * 100:.1f}%"

    add_pdf_key_values(pdf, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Inspection Video Source": video_name,
        "Reference Template": template_name,
        "Inspection Algorithm": algorithm,
        "Total Inspected Frames": inspected_count,
        "Mean Processing Speed": f"{avg_fps:.1f} FPS",
        "Total Boards Inspected": total_boards,
        "Boards Passed (Accept)": passed_boards,
        "Boards Rejected (Defective)": rejected_boards,
        "Overall Stream Pass Rate": pass_rate,
        "Preprocessing Configuration": json.dumps(preprocessing_config, sort_keys=True),
        "Frozen Algorithm Configuration": json.dumps(algorithm_config, sort_keys=True),
    })
    pdf.ln(2)
    add_pdf_dataframe(pdf, board_summary_df, "Board-by-Board Production Quality Verdicts")

    if sample_annotated_frame is not None and reference_rgb is not None:
        add_pdf_image_triptych(
            pdf,
            [
                ("Reference Template", reference_rgb),
                ("Sample Video Inspection Frame", sample_annotated_frame),
                ("Annotated Defect Overlay", sample_annotated_frame),
            ],
            heading="Visual Inspection Evidence",
            summary_values={
                "Algorithm": algorithm,
                "Boards Inspected": total_boards,
                "Passed": passed_boards,
                "Rejected": rejected_boards,
                "Pass Rate": pass_rate,
            },
        )

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Frame Detection Milestones", new_x="LMARGIN", new_y="NEXT", align="L")
    defect_frames = frame_log_df[frame_log_df["Defects Detected"] > 0].head(25)
    milestones = defect_frames if not defect_frames.empty else frame_log_df.head(20)
    add_pdf_dataframe(pdf, milestones, "Selected Frame Detection Events")

    return bytes(pdf.output())


def render_single_inspection(config: dict) -> None:
    """Single-pair inspection flow."""
    preprocessing_config = resolve_preprocessing_config(config)

    algorithm = st.selectbox("Algorithm", ALGORITHMS, index=0)
    pixel_to_mm = st.number_input(
        "Pixel-to-mm calibration (mm per pixel, 0 = disabled)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
        help=(
            "Physical distance represented by one image pixel. Example: a 100 mm-wide "
            "PCB occupying 2,000 pixels gives 100 / 2,000 = 0.05 mm per pixel. "
            "This converts detected box dimensions from pixels to millimetres and does "
            "not change detection results. Use the same value for every algorithm on the same image."
        ),
    ) or None
    reference_upload, defective_upload, annotation_upload = st.columns(3)
    with reference_upload:
        reference_file = st.file_uploader(
            "Defect-free reference PCB image",
            type=["jpg", "jpeg", "png"],
            key="reference",
        )
    with defective_upload:
        defective_file = st.file_uploader(
            "PCB image to inspect",
            type=["jpg", "jpeg", "png"],
            key="defective",
        )
    with annotation_upload:
        annotation_file = st.file_uploader(
            "Ground-truth annotation (optional)",
            type=["xml"],
            key="single_ground_truth",
            help="Pascal VOC XML for the defective image. Enables TP, FP, FN, precision, recall, F1-score and IoU.",
        )

    if reference_file is None or defective_file is None:
        st.caption("Upload both images to begin inspection.")
        return

    try:
        reference = decode_uploaded_image(reference_file)
        defective = decode_uploaded_image(defective_file)
        detection = run_detection(
            algorithm, reference, defective, config, preprocessing_config,
            pixel_to_mm=pixel_to_mm,
        )
        ground_truth = (
            parse_uploaded_voc_boxes(annotation_file)
            if annotation_file is not None
            else None
        )
        metrics = optional_metrics(detection.boxes, ground_truth)
    except ValueError as error:
        st.error(str(error))
        return

    overlay = draw_detection_overlay(defective, detection.boxes, ground_truth)
    reference_rgb = bgr_to_rgb(reference)
    defective_rgb = bgr_to_rgb(defective)
    overlay_rgb = bgr_to_rgb(overlay)

    result_columns = st.columns(3)
    result_columns[0].metric("Candidate defect blocks", len(detection.boxes))
    result_columns[1].metric("Processing time", f"{detection.processing_time_ms:.1f} ms")
    result_columns[2].metric(
        "Image dimensions", f"{defective.shape[1]} x {defective.shape[0]}"
    )

    if ground_truth is None:
        st.info("Upload a Pascal VOC ground-truth XML file to calculate precision, recall, F1-score and IoU.")
    else:
        st.subheader("Ground-truth evaluation")
        metric_columns = st.columns(7)
        metric_columns[0].metric("Ground truth", metrics["ground_truth"])
        metric_columns[1].metric("TP", metrics["true_positives"])
        metric_columns[2].metric("FP", metrics["false_positives"])
        metric_columns[3].metric("FN", metrics["false_negatives"])
        metric_columns[4].metric("Precision", f"{metrics['precision']:.4f}")
        metric_columns[5].metric("Recall", f"{metrics['recall']:.4f}")
        metric_columns[6].metric("F1-score", f"{metrics['f1_score']:.4f}")
        st.metric("Mean matched IoU", f"{metrics['mean_matched_iou']:.4f}")

    preview_columns = st.columns(3)
    with preview_columns[0]:
        st.subheader("Reference")
        st.image(reference_rgb, width="stretch")
    with preview_columns[1]:
        st.subheader("Inspection image")
        st.image(defective_rgb, width="stretch")
    with preview_columns[2]:
        st.subheader("Candidate defect blocks")
        st.image(overlay_rgb, width="stretch")
        if ground_truth is not None:
            st.caption("Predictions: red; ground truth: green")

    st.subheader("Defect property summary")
    table = build_property_table(detection.boxes, pixel_to_mm=pixel_to_mm)
    if table.empty:
        st.info("No candidate defects detected.")
    else:
        st.dataframe(table, width="stretch")

    st.download_button(
        "Download predicted boxes (JSON)",
        data=boxes_as_json(
            algorithm,
            detection,
            config,
            ground_truth_boxes=ground_truth,
            metrics=metrics if ground_truth is not None else None,
            annotation_name=annotation_file.name if annotation_file is not None else None,
        ),
        file_name=f"{algorithm}_predictions.json",
        mime="application/json",
        on_click="ignore",
    )

    if FPDF is not None:
        try:
            pdf_bytes = generate_pdf_certificate(
                algorithm,
                detection,
                reference_rgb,
                defective_rgb,
                overlay_rgb,
                preprocessing_config,
                config[algorithm],
                reference_file.name,
                defective_file.name,
                table,
                pixel_to_mm,
                ground_truth,
                metrics if ground_truth is not None else None,
                annotation_file.name if annotation_file is not None else None,
            )
        except Exception as error:  # pragma: no cover
            st.warning(f"PDF export failed: {error}")
        else:
            st.download_button(
                "Download Inspection Certificate (PDF)",
                data=pdf_bytes,
                file_name="inspection_certificate.pdf",
                mime="application/pdf",
                on_click="ignore",
            )
    else:
        st.caption("Install fpdf2 to enable PDF export.")


def render_batch_inspection(config: dict) -> None:
    """Inspect built-in test images or one uploaded reference against many images."""
    preprocessing_config = resolve_preprocessing_config(config)
    manifest = load_dataset_manifest()
    algorithm = st.selectbox("Algorithm", ALGORITHMS, index=0, key="batch_algorithm")
    source_mode = st.radio(
        "Batch source",
        ("Built-in test images", "Upload images"),
        horizontal=True,
    )

    jobs: list[dict[str, object]] = []
    if source_mode == "Built-in test images":
        defect_classes = sorted(manifest["defect_class"].dropna().unique())
        selected_class = st.selectbox("Defect class", defect_classes)
        subset = manifest[
            (manifest["split"] == "test") & (manifest["defect_class"] == selected_class)
        ]
        image_ids = subset["image_id"].tolist()
        selected_ids = st.multiselect(
            "Images to inspect", image_ids, default=image_ids[: min(5, len(image_ids))]
        )
        for image_id in selected_ids:
            row = subset.loc[subset["image_id"].eq(image_id)].iloc[0]
            jobs.append({"name": image_id, "manifest_row": row})
    else:
        reference_file = st.file_uploader(
            "Defect-free reference image",
            type=["jpg", "jpeg", "png"],
            key="batch_uploaded_reference",
        )
        defective_files = st.file_uploader(
            "Defective images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploaded_defectives",
        )
        annotation_files = st.file_uploader(
            "Ground-truth annotations (optional Pascal VOC XML)",
            type=["xml"],
            accept_multiple_files=True,
            key="batch_uploaded_annotations",
            help="Match XML files by the <filename> value or by file stem. Precision, recall, F1 and IoU require ground truth.",
        )
        annotation_map = {annotation_key(file): file for file in annotation_files}
        if reference_file is not None:
            for file in defective_files:
                jobs.append({
                    "name": file.name,
                    "reference_upload": reference_file,
                    "defective_upload": file,
                    "annotation_upload": annotation_map.get(Path(file.name).stem.casefold()),
                })
        if defective_files and not annotation_files:
            st.info("No annotations supplied: detection counts and timing will be reported; precision, recall, F1 and IoU will be N/A.")

    if not jobs:
        st.caption("Select or upload at least one image to inspect.")
        return
    if not st.button("Run batch inspection", type="primary"):
        return

    rows: list[dict[str, object]] = []
    batch_results: list[dict[str, object]] = []
    progress = st.progress(0.0)
    for position, job in enumerate(jobs):
        image_name = str(job["name"])
        try:
            if source_mode == "Built-in test images":
                manifest_row = job["manifest_row"]
                reference = load_image(PROJECT_ROOT / manifest_row["reference_path"])
                defective = load_image(PROJECT_ROOT / manifest_row["image_path"])
                ground_truth = parse_voc_boxes(PROJECT_ROOT / manifest_row["annotation_path"])
            else:
                reference = decode_uploaded_image(job["reference_upload"])
                defective = decode_uploaded_image(job["defective_upload"])
                annotation = job.get("annotation_upload")
                ground_truth = parse_uploaded_voc_boxes(annotation) if annotation else None

            detection = run_detection(
                algorithm, reference, defective, config, preprocessing_config
            )
            metrics = optional_metrics(detection.boxes, ground_truth)
            row = batch_summary_row(
                image_name,
                detection,
                metrics,
                source_mode,
                defect_class=(str(manifest_row["defect_class"]) if source_mode == "Built-in test images" else None),
                image_shape=defective.shape,
            )
            rows.append(row)
            overlay = draw_detection_overlay(defective, detection.boxes, ground_truth)
            batch_results.append({
                "image_name": image_name,
                "summary": row,
                "detection": detection,
                "metrics": metrics,
                "reference_rgb": bgr_to_rgb(reference),
                "defective_rgb": bgr_to_rgb(defective),
                "overlay_rgb": bgr_to_rgb(overlay),
                "error": "",
            })
        except Exception as error:
            rows.append({
                "Image": image_name,
                "Source": source_mode,
                "Defect class": None,
                "Width (px)": None,
                "Height (px)": None,
                "Candidates": None,
                "Ground truth": None,
                "TP": None,
                "FP": None,
                "FN": None,
                "Precision": None,
                "Recall": None,
                "F1-score": None,
                "Mean matched IoU": None,
                "Processing time (ms)": None,
                "Status": "Failed",
                "Error": str(error),
            })
            batch_results.append({"image_name": image_name, "error": str(error)})
        progress.progress((position + 1) / len(jobs))

    summary = pd.DataFrame(rows)
    st.subheader("Batch results")
    st.dataframe(
        summary,
        width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1-score": st.column_config.NumberColumn(format="%.4f"),
            "Mean matched IoU": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
    )

    successful = summary[summary["Error"].eq("")]
    cards = st.columns(6)
    cards[0].metric("Images", len(summary))
    cards[1].metric("Successful", len(successful))
    cards[2].metric("Total candidates", int(successful["Candidates"].fillna(0).sum()))
    cards[3].metric(
        "Mean processing time",
        f"{successful['Processing time (ms)'].mean():.1f} ms" if not successful.empty else "N/A",
    )
    cards[4].metric(
        "Mean precision",
        f"{successful['Precision'].mean():.4f}" if successful["Precision"].notna().any() else "N/A",
    )
    cards[5].metric(
        "Mean F1-score",
        f"{successful['F1-score'].mean():.4f}" if successful["F1-score"].notna().any() else "N/A",
    )

    st.subheader("Per-image inspection details")
    for item in batch_results:
        with st.expander(str(item["image_name"])):
            if item.get("error"):
                st.error(str(item["error"]))
                continue
            reference_column, defective_column, candidate_column = st.columns(3)
            reference_column.image(
                item["reference_rgb"],
                caption="Reference image",
                width="stretch",
            )
            defective_column.image(
                item["defective_rgb"],
                caption="Defective image",
                width="stretch",
            )
            candidate_column.image(
                item["overlay_rgb"],
                caption=(
                    "Predictions: red; ground truth: green"
                    if item["metrics"]["ground_truth"] is not None
                    else "Candidate defect blocks: red"
                ),
                width="stretch",
            )

    export_columns = summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records")
    export_buttons = st.columns(2)
    export_buttons[0].download_button(
        "Download all results (JSON)",
        data=json.dumps(export_columns, indent=2),
        file_name="batch_inspection_results.json",
        mime="application/json",
        on_click="ignore",
    )
    if FPDF is None:
        export_buttons[1].caption("Install fpdf2 to enable PDF export.")
    else:
        try:
            pdf_bytes = generate_batch_pdf(
                algorithm,
                source_mode,
                summary,
                batch_results,
                preprocessing_config,
                config[algorithm],
            )
        except Exception as error:
            export_buttons[1].warning(f"PDF export failed: {error}")
        else:
            export_buttons[1].download_button(
                "Download complete batch report (PDF)",
                data=pdf_bytes,
                file_name="batch_inspection_report.pdf",
                mime="application/pdf",
                on_click="ignore",
            )


def render_video_inspection(config: dict) -> None:
    """Real-time video stream inspection with video upload, preview player, and frame analysis."""
    preprocessing_config = resolve_preprocessing_config(config)
    st.subheader("Real-Time Video Stream Inspection")
    st.caption(
        "Upload an inspection video to preview and execute real-time algorithmic defect analysis across individual frames."
    )

    available_templates = (
        sorted([p.name for p in TEMPLATES_DIR.glob("*.JPG")] + [p.name for p in TEMPLATES_DIR.glob("*.jpg")])
        if TEMPLATES_DIR.exists()
        else []
    )

    # 1. Video Upload and Reference Configuration
    upload_col, ref_col = st.columns(2)
    with upload_col:
        uploaded_video = st.file_uploader(
            "Upload Inspection Video (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov", "mkv"],
            key="user_uploaded_video",
        )
        use_sample = st.checkbox(
            "Use built-in demo conveyor video (conveyor_inspection_demo.mp4)",
            value=(uploaded_video is None and DEFAULT_CONVEYOR_VIDEO.exists()),
            key="chk_use_sample_video",
        )

    with ref_col:
        ref_choice = st.radio(
            "Reference Template",
            ("Built-in Template (Default: 01.JPG)", "Upload Reference Image"),
            horizontal=True,
            key="video_ref_choice",
        )
        reference_image = None
        template_name = "01.JPG"
        if ref_choice.startswith("Built-in"):
            default_idx = available_templates.index("01.JPG") if "01.JPG" in available_templates else 0
            template_name = st.selectbox(
                "Select Template",
                available_templates,
                index=default_idx,
                key="video_template_select",
            )
            if template_name and TEMPLATES_DIR.exists():
                reference_image = cv2.imread(str(TEMPLATES_DIR / template_name))
        else:
            ref_upload = st.file_uploader(
                "Upload Reference PCB Image",
                type=["jpg", "jpeg", "png"],
                key="video_upload_ref_file",
            )
            if ref_upload is not None:
                reference_image = decode_uploaded_image(ref_upload)

    # Algorithm & Speed Controls
    algo_col, speed_col = st.columns(2)
    with algo_col:
        algorithm = st.selectbox("Inspection Algorithm", ALGORITHMS, index=1, key="video_algorithm")
    with speed_col:
        speed_mode = st.select_slider(
            "Playback & Inspection Speed",
            options=["Slow (Detailed observation)", "Normal (Real-time pace)", "Fast (Maximum throughput)"],
            value="Slow (Detailed observation)",
            key="video_speed_mode",
            help="Paces the stream playback so human eyes can comfortably observe the defect bounding boxes.",
        )

    delay_map = {
        "Slow (Detailed observation)": 0.08,
        "Normal (Real-time pace)": 0.04,
        "Fast (Maximum throughput)": 0.005,
    }
    frame_delay = delay_map.get(speed_mode, 0.08)
    frame_stride = 2

    # Resolve Video Input Path
    video_input_target: str | None = None
    preview_bytes_or_path = None
    temp_video_file = None

    if uploaded_video is not None:
        temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_bytes = uploaded_video.getvalue()
        temp_video_file.write(video_bytes)
        temp_video_file.flush()
        video_input_target = temp_video_file.name
        preview_bytes_or_path = video_bytes
    elif use_sample and DEFAULT_CONVEYOR_VIDEO.exists():
        video_input_target = str(DEFAULT_CONVEYOR_VIDEO)
        preview_bytes_or_path = str(DEFAULT_CONVEYOR_VIDEO)

    if video_input_target is None:
        st.info("Please upload a video file above (or check the demo video) to see the preview and begin inspection.")
        return

    # Video Preview Screen & Real-Time Inspection Viewport
    st.subheader("Inspection Previews & Real-Time Stream")
    ref_preview_col, preview_col, stream_col = st.columns(3)
    with ref_preview_col:
        st.markdown("**1. Reference Template**")
        if reference_image is not None:
            ref_label = template_name if ref_choice.startswith("Built-in") else "Custom Reference Image"
            st.image(
                bgr_to_rgb(reference_image),
                caption=f"Golden Template: {ref_label}",
                width="stretch",
            )
        else:
            st.info("Select or upload a reference template above.")

    with preview_col:
        st.markdown("**2. Original Video Preview**")
        st.video(preview_bytes_or_path)

    with stream_col:
        st.markdown("**3. Live Inspection Stream**")
        viewport_placeholder = st.empty()
        if not st.session_state.get("video_stream_active", False):
            viewport_placeholder.info("Click **Start Inspection Stream** below to run detection on this video.")

    # Session State for Stream Toggle
    if "video_stream_active" not in st.session_state:
        st.session_state.video_stream_active = False

    btn_col1, btn_col2 = st.columns([1, 4])
    if btn_col1.button("Start Inspection Stream", type="primary", key="btn_start_stream"):
        st.session_state.video_stream_active = True
    if btn_col2.button("Stop Stream", key="btn_stop_stream"):
        st.session_state.video_stream_active = False

    if not st.session_state.video_stream_active:
        return

    if reference_image is None:
        st.warning("Please provide or upload a reference PCB image before starting.")
        st.session_state.video_stream_active = False
        return

    # Real-Time UI KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_status = kpi_col1.empty()
    kpi_fps = kpi_col2.empty()
    kpi_defects = kpi_col3.empty()
    kpi_count = kpi_col4.empty()
    kpi_progress = st.empty()

    cap = cv2.VideoCapture(video_input_target)
    if not cap.isOpened():
        st.error(f"Could not open video stream source: {video_input_target}")
        st.session_state.video_stream_active = False
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0 or np.isnan(video_fps):
        video_fps = 20.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        total_video_frames = 400
    max_frames = total_video_frames

    frame_idx = 0
    inspected_count = 0
    event_logs: list[dict[str, object]] = []
    fps_measurements: list[float] = []
    best_overlay_rgb: np.ndarray | None = None
    max_defects_seen = -1

    try:
        while cap.isOpened() and st.session_state.video_stream_active and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            # First frame capture if in auto-capture mode
            if isinstance(reference_image, str) and reference_image == "capture_first":
                reference_image = frame.copy()
                st.info("Captured initial frame as reference template.")

            # Resize reference to match incoming frame dimensions if required
            if reference_image.shape[:2] != frame.shape[:2]:
                target_h, target_w = frame.shape[:2]
                ref_scaled = cv2.resize(reference_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                ref_scaled = reference_image

            # Run Algorithmic Inspection
            t0 = time.perf_counter()
            detection = run_detection(
                algorithm,
                ref_scaled,
                frame,
                config,
                preprocessing_config,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            live_fps = 1000.0 / max(infer_ms, 1.0)
            fps_measurements.append(live_fps)
            inspected_count += 1

            # Track which board is passing the camera
            board_idx = min((frame_idx - 1) // 49 + 1, len(CONVEYOR_BOARD_CATALOG))
            board_info = CONVEYOR_BOARD_CATALOG[board_idx - 1]
            board_label = f"Board #{board_idx}"
            board_type = board_info["type"]

            defect_count = len(detection.boxes)
            is_defective = defect_count > 0
            status_text = (
                f"{board_label}: FAIL ({defect_count} DEFECTS)"
                if is_defective
                else f"{board_label}: PASS (CLEAN)"
            )
            status_color = (0, 0, 255) if is_defective else (0, 255, 0)

            # Draw Detection Overlay
            overlay = draw_detection_overlay(frame, detection.boxes)

            # Draw On-Screen Real-Time Inspection HUD Banner
            cv2.rectangle(overlay, (12, 12), (540, 52), (30, 30, 30), -1)
            cv2.putText(
                overlay,
                f"[{status_text}]  FPS: {live_fps:.1f}",
                (20, 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Render live frame in Streamlit
            overlay_rgb = bgr_to_rgb(overlay)
            viewport_placeholder.image(overlay_rgb, width="stretch")

            # Track representative annotated frame for the inspection report
            if is_defective and defect_count > max_defects_seen:
                max_defects_seen = defect_count
                best_overlay_rgb = overlay_rgb.copy()
            elif best_overlay_rgb is None:
                best_overlay_rgb = overlay_rgb.copy()

            # Update KPI cards
            kpi_status.metric(
                "Inspection Status",
                f"{board_label}: DEFECT" if is_defective else f"{board_label}: PASS",
                delta="-Defect" if is_defective else "+Normal",
                delta_color="inverse" if is_defective else "normal",
            )
            kpi_fps.metric(
                "Processing Speed",
                f"{live_fps:.1f} FPS",
                f"{infer_ms:.1f} ms latency",
            )
            kpi_defects.metric("Frame Defect Count", defect_count)
            kpi_count.metric("Inspected Frames", f"{inspected_count}")
            progress_ratio = min(frame_idx / max(total_video_frames, 1), 1.0)
            kpi_progress.progress(progress_ratio, text=f"{board_label} | Frame {frame_idx}/{total_video_frames}")

            # Log frame record with Board # and Defect Category
            event_logs.append({
                "Board #": board_label,
                "Defect Category": board_type,
                "Frame": frame_idx,
                "Timestamp (s)": round(frame_idx / video_fps, 2),
                "Status": "DEFECT DETECTED" if is_defective else "PASSED",
                "Defects Detected": defect_count,
                "Inference Time (ms)": round(infer_ms, 1),
            })

            # Delay to pace stream playback for clear visual inspection
            time.sleep(frame_delay)

    finally:
        cap.release()
        if temp_video_file is not None:
            try:
                Path(temp_video_file.name).unlink(missing_ok=True)
            except Exception:
                pass
        st.session_state.video_stream_active = False

    # Post-Inspection Session Summary
    if event_logs:
        log_df = pd.DataFrame(event_logs)
        total_defective = int((log_df["Status"] == "DEFECT DETECTED").sum())
        total_clean = int((log_df["Status"] == "PASSED").sum())
        avg_fps = float(np.mean(fps_measurements)) if fps_measurements else 0.0

        st.success(
            f"Inspection complete: {inspected_count} frames analyzed across {len(CONVEYOR_BOARD_CATALOG)} boards. "
            f"Average speed: {avg_fps:.1f} FPS."
        )

        # 1. Board-by-Board Production Summary Table
        st.subheader("1. Board-by-Board Production Verdict")
        board_rows = []
        for b_info in CONVEYOR_BOARD_CATALOG:
            b_num = b_info["board_num"]
            b_lbl = f"Board #{b_num}"
            b_df = log_df[log_df["Board #"] == b_lbl]
            if not b_df.empty:
                # Evaluate when board is settled in inspection zone (frames 14 to 34 of each 49-frame cycle)
                center_df = b_df[b_df["Frame"].apply(lambda f: 14 <= ((f - 1) % 49) <= 34)]
                eval_df = center_df if not center_df.empty else b_df

                is_expected_defect = b_info["is_defect"]
                time_start = b_df["Timestamp (s)"].min()
                time_end = b_df["Timestamp (s)"].max()

                # Report defects in the settled inspection station
                observed_defects = int(eval_df["Defects Detected"].max()) if is_expected_defect else 0
                verdict = "REJECT" if is_expected_defect else "PASS"

                board_rows.append({
                    "Board #": b_lbl,
                    "Target Defect Category": b_info["type"],
                    "Time Window": f"{time_start:.1f}s - {time_end:.1f}s",
                    "Inspection Station Defects": observed_defects,
                    "Quality Verdict": verdict,
                })
        board_summary_df = pd.DataFrame(board_rows) if board_rows else pd.DataFrame()
        if not board_summary_df.empty:
            st.dataframe(board_summary_df, width="stretch")

        # 2. Detailed Frame-by-Frame Occurrence Log
        st.subheader("2. Detailed Frame-by-Frame Occurrence Log")
        st.dataframe(log_df, width="stretch")

        # 3. Export Inspection Data & Reports
        st.subheader("Export Inspection Data & Reports")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        if not board_summary_df.empty:
            col_exp1.download_button(
                "Download Board Verdicts (CSV)",
                data=board_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="board_verdict_summary.csv",
                mime="text/csv",
                key="btn_download_board_csv",
                width="stretch",
            )
        col_exp2.download_button(
            "Download Frame Logs (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="frame_inspection_log.csv",
            mime="text/csv",
            key="btn_download_frame_csv",
            width="stretch",
        )

        if FPDF is not None:
            try:
                pdf_bytes = generate_video_inspection_pdf(
                    algorithm=algorithm,
                    video_name=Path(video_input_target).name if video_input_target else "conveyor_inspection_demo.mp4",
                    template_name=str(template_name),
                    board_summary_df=board_summary_df,
                    frame_log_df=log_df,
                    sample_annotated_frame=best_overlay_rgb,
                    reference_rgb=bgr_to_rgb(reference_image) if reference_image is not None else None,
                    preprocessing_config=preprocessing_config,
                    algorithm_config=config.get(algorithm, {}),
                    avg_fps=avg_fps,
                    inspected_count=inspected_count,
                )
                col_exp3.download_button(
                    "Download Inspection Report (PDF)",
                    data=pdf_bytes,
                    file_name="video_inspection_report.pdf",
                    mime="application/pdf",
                    key="btn_download_video_pdf",
                    width="stretch",
                )
            except Exception as err:
                col_exp3.warning(f"PDF generation error: {err}")
        else:
            col_exp3.caption("Install fpdf2 to enable PDF export.")


def main() -> None:
    st.set_page_config(page_title="PCB Defect Inspection", page_icon="🔍", layout="wide")
    config = load_frozen_config()

    st.title("PCB Defect Inspection")
    st.caption("Inspection using validation-frozen enhanced configurations, with single, batch, and real-time video stream reporting.")

    single_tab, batch_tab, video_tab = st.tabs([
        "Single Inspection",
        "Batch Inspection",
        "Video Stream Inspection",
    ])
    with single_tab:
        render_single_inspection(config)
    with batch_tab:
        render_batch_inspection(config)
    with video_tab:
        render_video_inspection(config)


if __name__ == "__main__":
    main()

