from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def validate_pair(reference: np.ndarray, defective: np.ndarray) -> None:
    """Validate a reference/defective pair without altering either image."""
    if reference is None or defective is None:
        raise ValueError("Reference and defective images are required.")

    if reference.ndim not in (2, 3) or defective.ndim not in (2, 3):
        raise ValueError("Images must be grayscale or color arrays.")

    reference_size = reference.shape[:2]
    defective_size = defective.shape[:2]
    if reference_size != defective_size:
        raise ValueError(
            "Reference and defective image dimensions must match; "
            f"got {reference_size} and {defective_size}. "
            "Automatic resizing is disabled because it can invalidate XML coordinates."
        )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale image without denoising or enhancement."""
    if image is None:
        raise ValueError("An image is required.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def preprocess_pair(
    reference: np.ndarray,
    defective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the neutral preprocessing shared by all raw algorithms.

    Algorithm-specific representations such as absolute differences, edge maps,
    similarity responses, and descriptors must be created by each algorithm.
    """
    validate_pair(reference, defective)
    return to_grayscale(reference), to_grayscale(defective)
