#!/usr/bin/env python3
"""Convert the local annotation-kit JSON to YOLO detection labels without copying images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
import math
from pathlib import Path
from math import isfinite

import yaml

RAW_CLASSES = ("boss", "animation", "projectile", "other")
BOSS_CLASS_BY_LABEL = {
    "Zulrah": "boss_zulrah",
    "Vorkath": "boss_vorkath",
    "Great Olm": "boss_great_olm",
    "Theatre of Blood": "boss_verzik",
    "Maggot King": "boss_maggot_king",
    "Yama": "boss_yama",
}
CLASSES = (
    "boss_zulrah",
    "boss_vorkath",
    "boss_great_olm",
    "boss_verzik",
    "boss_maggot_king",
    "boss_yama",
    "animation",
    "projectile",
    "other",
)
RATIOS = (0.70, 0.20, 0.10)


def split_random(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    n_train = int(len(items) * RATIOS[0])
    n_val = int(len(items) * RATIOS[1])
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}


def split_grouped(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    """Allocate whole source clips, keeping at least one training example of each class."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in samples:
        groups[item["source_file"]].append(item)
    if len(groups) < 3:
        raise ValueError("Clip-grouped 70/20/10 requires at least three independent source clips")

    rng = random.Random(seed)
    sources = list(groups)
    rng.shuffle(sources)
    source_classes = {
        source: {class_id for item in groups[source] for class_id, _ in item.get("rows", [])}
        for source in sources
    }
    all_classes = set().union(*source_classes.values()) if source_classes else set()
    train_sources: set[str] = set()
    uncovered = set(all_classes)
    while uncovered:
        candidates = [source for source in sources if source not in train_sources]
        if not candidates:
            raise ValueError("Could not retain every class in the training split")
        chosen = max(
            candidates,
            key=lambda source: (len(source_classes[source] & uncovered), len(groups[source])),
        )
        train_sources.add(chosen)
        uncovered.difference_update(source_classes[chosen])

    group_items = [(source, groups[source]) for source in sources if source not in train_sources]
    group_items.sort(key=lambda entry: len(entry[1]), reverse=True)
    target_counts = [len(samples) * ratio for ratio in RATIOS]
    split_names = ("train", "val", "test")
    allocations: list[list[tuple[str, list[dict]]]] = [[(source, groups[source]) for source in sources if source in train_sources], [], []]
    counts = [sum(len(groups[source]) for source in train_sources), 0, 0]

    for position, group in enumerate(group_items):
        remaining = len(group_items) - position
        empty = [i for i, bucket in enumerate(allocations) if not bucket]
        candidates = empty if remaining <= len(empty) else range(3)
        chosen = min(candidates, key=lambda i: (counts[i] + len(group[1]) - target_counts[i]) / max(target_counts[i], 1))
        allocations[chosen].append(group)
        counts[chosen] += len(group[1])

    return {
        split_names[i]: [sample for _, group in allocations[i] for sample in group]
        for i in range(3)
    }


def bbox_from_polygon(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Every polygon needs at least three points")
    coordinates = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Polygon points must be [x, y] pairs")
        x, y = float(point[0]), float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Polygon coordinates must be finite numbers")
        coordinates.append((min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))))
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def load_samples(packet: Path, labels_path: Path) -> list[dict]:
    labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    tasks = labels.get("tasks", [])
    if not tasks:
        raise ValueError("The label file has no tasks")
    quality = labels.get("quality_check", {})
    reviewed_ids = set(quality.get("second_review_frame_ids", []))
    required_review_count = math.ceil(len(tasks) * 0.20)
    if quality.get("draft") is not False or quality.get("frozen") is not True or quality.get("independent_review_complete") is not True:
        raise ValueError("Labels must be independently reviewed and frozen before dataset preparation")
    if len(reviewed_ids) < required_review_count or quality.get("second_review_fraction", 0) < 0.20:
        raise ValueError(f"At least {required_review_count} frames require independent second review")
    samples = []
    for task in tasks:
        ann = task.get("annotation", {})
        for field in ("boss_visibility", "event_visibility", "source_markup"):
            if ann.get(field, "unlabeled") in ("", "unlabeled", None):
                raise ValueError(f"{task.get('frame_id')} is missing reviewed {field}")
        if "uncertain" in (ann["boss_visibility"], ann["event_visibility"], ann["source_markup"]):
            raise ValueError(f"{task.get('frame_id')} still has an uncertain label; resolve it before training")
        if ann["boss_visibility"] not in ("visible", "not_visible"):
            raise ValueError(f"{task.get('frame_id')} has invalid boss_visibility")
        if ann["event_visibility"] not in ("visible", "not_visible"):
            raise ValueError(f"{task.get('frame_id')} has invalid event_visibility")
        if ann["source_markup"] not in ("none", "visible"):
            raise ValueError(f"{task.get('frame_id')} has invalid source_markup")
        image = (packet / task["image"]).resolve()
        if not image.is_relative_to(packet.resolve()):
            raise ValueError(f"{task.get('frame_id')} image path escapes the packet directory")
        if not image.is_file():
            raise FileNotFoundError(image)
        width, height = int(task["width"]), int(task["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"{task.get('frame_id')} has invalid image dimensions")
        rows = []
        raw_classes = set()
        for polygon in ann.get("polygons", []):
            raw_class = polygon.get("class")
            if raw_class not in RAW_CLASSES:
                raise ValueError(f"{task.get('frame_id')} has unknown polygon class {raw_class!r}")
            raw_classes.add(raw_class)
            if raw_class == "boss":
                cls = BOSS_CLASS_BY_LABEL.get(task.get("boss"))
                if cls is None:
                    raise ValueError(f"{task.get('frame_id')} has unsupported boss label {task.get('boss')!r}")
            else:
                cls = raw_class
            bbox = bbox_from_polygon(polygon.get("points", []), width, height)
            rows.append((CLASSES.index(cls), bbox))
        if ann["boss_visibility"] == "visible" and "boss" not in raw_classes:
            raise ValueError(f"{task.get('frame_id')} marks the boss visible without a boss polygon")
        if ann["boss_visibility"] == "not_visible" and "boss" in raw_classes:
            raise ValueError(f"{task.get('frame_id')} has a boss polygon marked not visible")
        event_classes = raw_classes.intersection(("animation", "projectile"))
        if ann["event_visibility"] == "visible" and not event_classes:
            raise ValueError(f"{task.get('frame_id')} marks an event visible without an animation/projectile polygon")
        if ann["event_visibility"] == "not_visible" and event_classes:
            raise ValueError(f"{task.get('frame_id')} has an event polygon marked not visible")
        samples.append({
            "frame_id": task["frame_id"],
            "image": image,
            "source_file": task["source_file"],
            "source_seconds": task["source_seconds"],
            "width": width,
            "height": height,
            "rows": rows,
        })
    if not samples:
        raise ValueError("No reviewed, non-uncertain frames remain")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("random", "grouped"), required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--overwrite-label-files", action="store_true")
    args = parser.parse_args()

    packet = args.packet.resolve()
    output = args.output.resolve()
    samples = load_samples(packet, args.labels.resolve())
    buckets = split_random(samples, args.seed) if args.split == "random" else split_grouped(samples, args.seed)

    label_paths = [item["image"].with_suffix(".txt") for item in samples]
    conflicts = [path for path in label_paths if path.exists()]
    if conflicts and not args.overwrite_label_files:
        raise FileExistsError(
            f"Refusing to overwrite {len(conflicts)} existing label files; inspect them or pass "
            "--overwrite-label-files after confirming they belong to this dataset."
        )

    output.mkdir(parents=True, exist_ok=True)
    for item, label_path in zip(samples, label_paths):
        content = "".join(
            f"{class_id} " + " ".join(f"{value:.8f}" for value in bbox) + "\n"
            for class_id, bbox in item["rows"]
        )
        label_path.write_text(content, encoding="utf-8")

    for name, rows in buckets.items():
        (output / f"{name}.txt").write_text("\n".join(str(row["image"]) for row in rows) + "\n", encoding="utf-8")

    data_yaml = {"path": str(output), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": list(CLASSES)}
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    split_doc = {
        "split": args.split,
        "seed": args.seed,
        "ratio_target": dict(zip(("train", "val", "test"), RATIOS)),
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "source_clips": {name: sorted({row["source_file"] for row in rows}) for name, rows in buckets.items()},
        "class_box_counts": {
            name: dict(Counter(CLASSES[class_id] for row in rows for class_id, _ in row["rows"]))
            for name, rows in buckets.items()
        },
        "warning": "Frame-random splits can leak adjacent video frames. Use grouped as the primary result.",
    }
    (output / "split_manifest.json").write_text(json.dumps(split_doc, indent=2), encoding="utf-8")
    print(json.dumps(split_doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
