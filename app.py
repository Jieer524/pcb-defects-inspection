"""Streamlit deployment: defect inspection with property table, PDF export, batch mode."""

from __future__ import annotations

import json
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
            calibrate=bool(section.get("calibrate", True)),
            ransac_reproj_threshold=float(section.get("ransac_reproj_threshold", 5.0)),
            preprocessing_config=preprocessing_config,
            merge_points=bool(section.get("merge_points", True)),
            morph_dilate=int(section.get("morph_dilate", 0)),
            min_area=float(section.get("min_area", 300.0)),
        )
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def draw_detection_overlay(defective_image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Draw all predicted boxes on a copy of the defective image."""
    overlay = defective_image.copy()
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


def boxes_as_json(algorithm: str, detection, config: dict) -> str:
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
    }
    return json.dumps(payload, indent=2)


def generate_pdf_certificate(
    algorithm: str,
    detection,
    reference_rgb: np.ndarray,
    defective_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    preprocessing_config: dict,
) -> bytes:
    """Generate an inspection certificate PDF using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PCB Defect Inspection Certificate", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Algorithm: {algorithm}", ln=True)
    pdf.cell(
        0,
        6,
        f"Image dimensions: {detection.reference_gray.shape[1]} x {detection.reference_gray.shape[0]} px",
        ln=True,
    )
    pdf.cell(0, 6, f"Preprocessing: {json.dumps(preprocessing_config)}", ln=True)
    pdf.cell(0, 6, f"Total candidates: {len(detection.boxes)}", ln=True)
    pdf.cell(0, 6, f"Processing time: {detection.processing_time_ms:.1f} ms", ln=True)

    status = "DEFECTS DETECTED" if detection.boxes else "NO DEFECTS DETECTED"
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Status: {status}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Defect Property Summary", ln=True)
    pdf.set_font("Helvetica", "", 8)
    table = build_property_table(detection.boxes)
    max_table_rows = 1000
    if len(table) > max_table_rows:
        pdf.cell(0, 6, f"(showing first {max_table_rows} of {len(table)} candidates)", ln=True)
    table = table.head(max_table_rows)
    column_widths = [14, 16, 16, 16, 16, 18, 18, 16, 18, 20]
    headers = list(table.columns)[: len(column_widths)]
    for header, width in zip(headers, column_widths):
        pdf.cell(width, 6, header, border=1)
    pdf.ln()
    for _, row in table.iterrows():
        for value, width in zip(row[: len(column_widths)], column_widths):
            pdf.cell(width, 6, str(value), border=1)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Inspection Images", ln=True)

    for label, image_rgb in (
        ("Reference", reference_rgb),
        ("Defective", defective_rgb),
        ("Overlay", overlay_rgb),
    ):
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, label, ln=True)
        retval, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not retval or buffer is None:
            raise ValueError(f"Could not encode image for PDF ({label}).")
        with BytesIO(buffer.tobytes()) as buf:
            pdf.image(buf, w=160)
        pdf.ln(2)

    footer = "PCB Defect Inspection Certificate - end of report"
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, footer, align="C")

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
    ) or None
    reference_upload, defective_upload = st.columns(2)
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
    except ValueError as error:
        st.error(str(error))
        return

    overlay = draw_detection_overlay(defective, detection.boxes)
    reference_rgb = bgr_to_rgb(reference)
    defective_rgb = bgr_to_rgb(defective)
    overlay_rgb = bgr_to_rgb(overlay)

    result_columns = st.columns(3)
    result_columns[0].metric("Candidate defect blocks", len(detection.boxes))
    result_columns[1].metric("Processing time", f"{detection.processing_time_ms:.1f} ms")
    result_columns[2].metric(
        "Image dimensions", f"{defective.shape[1]} x {defective.shape[0]}"
    )

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

    st.subheader("Defect property summary")
    table = build_property_table(detection.boxes, pixel_to_mm=pixel_to_mm)
    if table.empty:
        st.info("No candidate defects detected.")
    else:
        st.dataframe(table, use_container_width=True)

    st.download_button(
        "Download predicted boxes (JSON)",
        data=boxes_as_json(algorithm, detection, config),
        file_name=f"{algorithm}_predictions.json",
        mime="application/json",
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
            )
        except Exception as error:  # pragma: no cover
            st.warning(f"PDF export failed: {error}")
        else:
            st.download_button(
                "Download Inspection Certificate (PDF)",
                data=pdf_bytes,
                file_name="inspection_certificate.pdf",
                mime="application/pdf",
            )
    else:
        st.caption("Install fpdf2 to enable PDF export.")


def render_batch_inspection(config: dict) -> None:
    """Batch / folder inspection flow using the dataset manifest."""
    preprocessing_config = resolve_preprocessing_config(config)
    manifest = load_dataset_manifest()

    algorithm = st.selectbox("Algorithm", ALGORITHMS, index=0, key="batch_algorithm")
    defect_classes = sorted(manifest["defect_class"].dropna().unique())
    selected_class = st.selectbox("Defect class", defect_classes)

    subset = manifest[
        (manifest["split"] == "test") & (manifest["defect_class"] == selected_class)
    ]
    if subset.empty:
        st.info("No test images for this class.")
        return

    image_ids = subset["image_id"].tolist()
    selected_ids = st.multiselect(
        "Images to inspect", image_ids, default=image_ids[: min(5, len(image_ids))]
    )

    if not selected_ids:
        st.caption("Select at least one image.")
        return

    run_button = st.button("Run batch inspection")
    if not run_button:
        return

    rows = []
    progress = st.progress(0.0)
    for position, image_id in enumerate(selected_ids):
        row = subset[subset["image_id"] == image_id].iloc[0]
        try:
            reference = load_image(PROJECT_ROOT / row["reference_path"])
            defective = load_image(PROJECT_ROOT / row["image_path"])
            detection = run_detection(
                algorithm, reference, defective, config, preprocessing_config
            )
            rows.append(
                {
                    "image_id": image_id,
                    "defect_class": row["defect_class"],
                    "candidates": len(detection.boxes),
                    "processing_time_ms": round(detection.processing_time_ms, 1),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "image_id": image_id,
                    "defect_class": row["defect_class"],
                    "candidates": None,
                    "processing_time_ms": None,
                    "error": str(error),
                }
            )
        progress.progress((position + 1) / len(selected_ids))

    summary = pd.DataFrame(rows)
    st.subheader("Batch summary")
    st.dataframe(summary, use_container_width=True)

    st.download_button(
        "Download All Results (JSON)",
        data=summary.to_json(indent=2, orient="records"),
        file_name="batch_inspection_results.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="PCB Defect Inspection", page_icon="🔍", layout="wide")
    config = load_frozen_config()

    st.title("PCB Defect Inspection")
    st.caption("Inspection with optional preprocessing, ORB calibration, and batch mode.")

    single_tab, batch_tab = st.tabs(["Single Inspection", "Batch Inspection"])
    with single_tab:
        render_single_inspection(config)
    with batch_tab:
        render_batch_inspection(config)


if __name__ == "__main__":
    main()
