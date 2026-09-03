"""Streamlit deployment: defect inspection with property table, PDF export, batch mode."""

from __future__ import annotations

import json
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

ALGORITHMS = ("template_matching", "otsu", "canny", "orb")


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
        st.image(reference_rgb, use_container_width=True)
    with preview_columns[1]:
        st.subheader("Inspection image")
        st.image(defective_rgb, use_container_width=True)
    with preview_columns[2]:
        st.subheader("Candidate defect blocks")
        st.image(overlay_rgb, use_container_width=True)
        if ground_truth is not None:
            st.caption("Predictions: red; ground truth: green")

    st.subheader("Defect property summary")
    table = build_property_table(detection.boxes, pixel_to_mm=pixel_to_mm)
    if table.empty:
        st.info("No candidate defects detected.")
    else:
        st.dataframe(table, use_container_width=True)

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
        use_container_width=True,
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
                use_container_width=True,
            )
            defective_column.image(
                item["defective_rgb"],
                caption="Defective image",
                use_container_width=True,
            )
            candidate_column.image(
                item["overlay_rgb"],
                caption=(
                    "Predictions: red; ground truth: green"
                    if item["metrics"]["ground_truth"] is not None
                    else "Candidate defect blocks: red"
                ),
                use_container_width=True,
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


def main() -> None:
    st.set_page_config(page_title="PCB Defect Inspection", page_icon="🔍", layout="wide")
    config = load_frozen_config()

    st.title("PCB Defect Inspection")
    st.caption("Inspection using validation-frozen enhanced configurations, with single and batch reporting.")

    single_tab, batch_tab = st.tabs(["Single Inspection", "Batch Inspection"])
    with single_tab:
        render_single_inspection(config)
    with batch_tab:
        render_batch_inspection(config)


if __name__ == "__main__":
    main()
