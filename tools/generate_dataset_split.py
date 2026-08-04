"""Generate the authoritative aligned PCB dataset split manifest."""

from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


SEED = 42
TARGETS = {"development": 139, "validation": 139, "test": 415}
CLASS_NAMES = {
    "Missing_hole": "missing_hole",
    "Mouse_bite": "mouse_bite",
    "Open_circuit": "open_circuit",
    "Short": "short_circuit",
    "Spur": "spur",
    "Spurious_copper": "spurious_copper",
}


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_records(project_root: Path) -> list[dict[str, str]]:
    dataset_root = project_root / "data" / "raw" / "PCB_DATASET"
    records: list[dict[str, str]] = []

    for folder, defect_class in CLASS_NAMES.items():
        image_dir = dataset_root / "images" / folder
        annotation_dir = dataset_root / "Annotations" / folder

        for image_path in sorted(image_dir.glob("*.jpg"), key=lambda p: p.name.lower()):
            annotation_path = annotation_dir / f"{image_path.stem}.xml"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation: {annotation_path}")

            xml_filename = ET.parse(annotation_path).findtext("filename")
            if xml_filename != image_path.name:
                raise ValueError(
                    f"Annotation filename mismatch for {annotation_path}: {xml_filename!r}"
                )

            board_id = image_path.stem.split("_", 1)[0]
            reference_path = dataset_root / "PCB_USED" / f"{board_id}.JPG"
            if not reference_path.is_file():
                raise FileNotFoundError(f"Missing reference: {reference_path}")

            records.append(
                {
                    "image_id": image_path.stem,
                    "image_path": relative(image_path, project_root),
                    "reference_path": relative(reference_path, project_root),
                    "annotation_path": relative(annotation_path, project_root),
                    "defect_class": defect_class,
                    # No duplicate/source-scene metadata is supplied. Each annotated
                    # aligned image is therefore the smallest defensible source group.
                    "group_id": image_path.stem,
                }
            )

    if len(records) != 693:
        raise ValueError(f"Expected 693 aligned records, found {len(records)}")
    if len({record["image_id"] for record in records}) != len(records):
        raise ValueError("image_id values are not unique")
    return records


def assign_splits(records: list[dict[str, str]]) -> None:
    by_class: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_class.setdefault(record["defect_class"], []).append(record)

    # Six classes receive 23 records in each tuning portion. The one extra
    # development and validation record is assigned to different 116-image
    # classes so every class remains within one image of every other class.
    development_extra = "open_circuit"
    validation_extra = "short_circuit"

    for class_index, defect_class in enumerate(sorted(by_class)):
        class_records = sorted(by_class[defect_class], key=lambda row: row["image_id"])
        random.Random(SEED + class_index).shuffle(class_records)
        development_count = 24 if defect_class == development_extra else 23
        validation_count = 24 if defect_class == validation_extra else 23

        for index, record in enumerate(class_records):
            if index < development_count:
                record["split"] = "development"
            elif index < development_count + validation_count:
                record["split"] = "validation"
            else:
                record["split"] = "test"

    counts = Counter(record["split"] for record in records)
    if counts != Counter(TARGETS):
        raise ValueError(f"Incorrect split counts: {dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_split.csv"),
        help="Output path relative to the project root",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    records = collect_records(project_root)
    assign_splits(records)
    records.sort(key=lambda row: row["image_id"])

    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "image_path",
        "reference_path",
        "annotation_path",
        "defect_class",
        "group_id",
        "split",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records to {output_path}")
    print(dict(Counter(record["split"] for record in records)))


if __name__ == "__main__":
    main()
