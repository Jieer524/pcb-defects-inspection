"""Generate an industrial conveyor inspection simulation video from PCB dataset images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "PCB_USED" / "01.JPG"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "conveyor_inspection_demo.mp4"

DEFAULT_DEFECT_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images" / "Missing_hole" / "01_missing_hole_01.jpg",
    PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images" / "Short" / "01_short_01.jpg",
    PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images" / "Mouse_bite" / "01_mouse_bite_01.jpg",
    PROJECT_ROOT / "data" / "raw" / "PCB_DATASET" / "images" / "Spurious_copper" / "01_spurious_copper_01.jpg",
]


def add_slight_jitter(frame: np.ndarray, max_shift: int = 2, max_angle: float = 0.2) -> np.ndarray:
    """Simulate small mechanical vibration on a conveyor belt."""
    h, w = frame.shape[:2]
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    angle = np.random.uniform(-max_angle, max_angle)

    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy

    return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def generate_conveyor_video(
    template_path: Path | str = DEFAULT_TEMPLATE_PATH,
    defect_paths: list[Path | str] | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_width: int = 1280,
    target_height: int = 670,
    fps: int = 20,
    frames_per_board_inspection: int = 25,
    frames_transition: int = 12,
) -> Path:
    """Synthesise an AOI conveyor belt video alternating between clean and defective boards."""
    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if defect_paths is None:
        defect_paths = DEFAULT_DEFECT_PATHS

    template_img = cv2.imread(str(template_path))
    if template_img is None:
        raise FileNotFoundError(f"Could not load reference template from {template_path}")
    template_resized = cv2.resize(template_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Prepare list of boards: clean -> defective1 -> clean -> defective2 -> ...
    boards: list[tuple[str, np.ndarray]] = [("Clean Board (Normal)", template_resized)]
    for p in defect_paths:
        p = Path(p)
        if not p.exists():
            continue
        defect_img = cv2.imread(str(p))
        if defect_img is not None:
            defect_resized = cv2.resize(defect_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            boards.append((f"Defective ({p.parent.name})", defect_resized))
            boards.append(("Clean Board (Normal)", template_resized))

    # Video Writer (prefer avc1 / H.264 for native browser playback)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Blank conveyor background (dark industrial gray)
    conveyor_bg = np.full((target_height, target_width, 3), 40, dtype=np.uint8)
    # Add subtle conveyor belt texture lines
    for y in range(0, target_height, 20):
        cv2.line(conveyor_bg, (0, y), (target_width, y), (48, 48, 48), 1)

    try:
        for b_idx, (label, board_img) in enumerate(boards):
            # 1. Slide board into view from left (-width to 0)
            for t in range(frames_transition):
                offset_x = int(target_width * (1.0 - np.sin((t / frames_transition) * (np.pi / 2))))
                frame = conveyor_bg.copy()
                if offset_x < target_width:
                    visible_w = target_width - offset_x
                    frame[:, :visible_w] = board_img[:, offset_x:]
                writer.write(frame)

            # 2. Board stationed under camera for inspection
            for _ in range(frames_per_board_inspection):
                writer.write(board_img)

            # 3. Slide board out of view to right (0 to target_width)
            for t in range(frames_transition):
                offset_x = int(target_width * np.sin((t / frames_transition) * (np.pi / 2)))
                frame = conveyor_bg.copy()
                if offset_x < target_width:
                    visible_w = target_width - offset_x
                    frame[:, offset_x:] = board_img[:, :visible_w]
                writer.write(frame)

    finally:
        writer.release()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic conveyor inspection video")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to output mp4")
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate")
    args = parser.parse_args()

    print(f"Generating conveyor video to {args.output} ...")
    out_file = generate_conveyor_video(output_path=args.output, fps=args.fps)
    print(f"Successfully generated conveyor video: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
