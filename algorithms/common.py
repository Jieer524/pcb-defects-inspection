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


def preprocess_pair(
    reference: np.ndarray,
    defective: np.ndarray,
    blur_kernel: tuple[int, int] = (5, 5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference is None or defective is None:
        raise ValueError("Reference and defective images are required.")

    height, width = reference.shape[:2]

    defective_resized = cv2.resize(
        defective,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    defective_gray = cv2.cvtColor(
        defective_resized,
        cv2.COLOR_BGR2GRAY
    )

    reference_blur = cv2.GaussianBlur(
        reference_gray,
        blur_kernel,
        0
    )

    defective_blur = cv2.GaussianBlur(
        defective_gray,
        blur_kernel,
        0
    )

    difference = cv2.absdiff(
        reference_blur,
        defective_blur
    )

    return reference_blur, defective_blur, difference