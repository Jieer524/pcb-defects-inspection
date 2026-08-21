"""Shared image preprocessing (denoising + contrast enhancement).

These operations are applied uniformly and optionally to all four algorithms so
that any comparison between them remains fair. By default every function is a
no-op fallback so that existing callers that pass no configuration keep the
original raw behaviour.
"""

from __future__ import annotations

import cv2
import numpy as np


def denoise(
    gray: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply a noise-suppression filter to an 8-bit grayscale image.

    Methods:
    - "gaussian": ``cv2.GaussianBlur`` with sigma=0 (auto-calculated from kernel)
    - "median": ``cv2.medianBlur`` for impulse (salt-and-pepper) noise
    - "none": returns the input unchanged
    """
    if gray.ndim != 2:
        raise ValueError("denoise expects a single-channel grayscale image.")

    method = method.lower()
    if method in ("none", ""):
        return gray
    if method == "gaussian":
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=0)
    if method == "median":
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        return cv2.medianBlur(gray, kernel_size)
    raise ValueError(f"Unknown denoise method '{method}'.")


def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Pass ``clip_limit`` <= 0 to bypass enhancement and return the input unchanged.
    """
    if gray.ndim != 2:
        raise ValueError("enhance_contrast expects a single-channel grayscale image.")
    if clip_limit <= 0 or tile_size <= 1:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def preprocess_image(gray: np.ndarray, config: dict | None = None) -> np.ndarray:
    """Apply the full preprocessing pipeline (denoise -> enhance) from a config dict.

    The config may contain any of the keys used in ``frozen_parameters.yaml``:

    - ``denoise_method``: "gaussian" | "median" | "none"
    - ``denoise_kernel_size``: positive odd integer
    - ``contrast_enhancement``: "clahe" | "none"
    - ``clahe_clip_limit``: float
    - ``clahe_tile_size``: integer > 1

    A ``None`` or empty config leaves the image unchanged.
    """
    if not config:
        return gray

    method = str(config.get("denoise_method", "none")).lower()
    if method not in ("none", ""):
        kernel_size = int(config.get("denoise_kernel_size", 3))
        gray = denoise(gray, method=method, kernel_size=kernel_size)

    enhancement = str(config.get("contrast_enhancement", "none")).lower()
    if enhancement not in ("none", ""):
        clip_limit = float(config.get("clahe_clip_limit", 2.0))
        tile_size = int(config.get("clahe_tile_size", 8))
        gray = enhance_contrast(gray, clip_limit=clip_limit, tile_size=tile_size)

    return gray


def build_preprocessing_config(section: dict | None) -> dict:
    """Translate the frozen YAML ``preprocessing`` section into the config dict
    consumed by the algorithms.

    Returns a no-op configuration (denoise off, contrast off) when the section is
    missing or ``enabled`` is false, preserving the original raw behaviour.
    """
    if not section or not section.get("enabled", False):
        return {"denoise_method": "none", "contrast_enhancement": "none"}
    return {
        "denoise_method": section.get("denoise_method", "none"),
        "denoise_kernel_size": section.get("denoise_kernel_size", 3),
        "contrast_enhancement": section.get("contrast_enhancement", "none"),
        "clahe_clip_limit": section.get("clahe_clip_limit", 2.0),
        "clahe_tile_size": section.get("clahe_tile_size", 8),
    }
