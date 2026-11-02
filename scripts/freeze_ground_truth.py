#!/usr/bin/env python3
"""Validate an independent 20% label review and freeze reviewed detector labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


VISIBILITY_FIELDS = ("boss_visibility", "event_visibility", "source_markup")
LABELED_VALUES = {"boss_visibility": {"visible", "not_visible"}, "event_visibility": {"visible", "not_visible"}, "source_markup": {"visible", "none"}}


def bbox(poly: dict) -> tuple[float, float, float, float]:
    points = poly.get("points", [])
    if len(points) < 3:
        raise ValueError("Each reviewed polygon needs at least three points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def annotation_complete(annotation: dict) -> bool:
    return all(annotation.get(field) in values for field, values in LABELED_VALUES.items())


def compare_boxes(first: list[dict], second: list[dict], threshold: float) -> list[dict]:
    conflicts = []
    classes = sorted({p.get("class") for p in first} | {p.get("class") for p in second})
    for cls in classes:
        a = [p for p in first if p.get("class") == cls]
        b = [p for p in second if p.get("class") == cls]
        if len(a) != len(b):
            conflicts.append({"class": cls, "first_count": len(a), "second_count": len(b), "reason": "object count differs"})
            continue
        unmatched = set(range(len(b)))
        for pa in a:
            candidates = [(iou(bbox(pa), bbox(b[j])), j) for j in unmatched]
            if not candidates:
                conflicts.append({"class": cls, "reason": "no matching polygon"})
                break
            score, index = max(candidates)
            if score < threshold:
                conflicts.append({"class": cls, "first_box": bbox(pa), "best_second_iou": round(score, 4), "required_iou": threshold, "reason": "box overlap below review threshold"})
            unmatched.remove(index)
    return conflicts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True, help="Complete first-pass labels JSON")
    parser.add_argument("--second", type=Path, required=True, help="Independent reviewer export JSON")
    parser.add_argument("--manifest", type=Path, required=True, help="Original frame_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Frozen label JSON output")
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()

    first_path, second_path, manifest_path = args.first.resolve(), args.second.resolve(), args.manifest.resolve()
    first = json.loads(first_path.read_text(encoding="utf-8-sig"))
    second = json.loads(second_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    first_tasks = {task["frame_id"]: task for task in first.get("tasks", [])}
    second_tasks = {task["frame_id"]: task for task in second.get("tasks", [])}
    frame_ids = [task["frame_id"] for task in manifest.get("tasks", [])]
    if not frame_ids or set(first_tasks) != set(frame_ids) or set(second_tasks) != set(frame_ids):
        parser.error("first labels, second review, and manifest must contain the same frame IDs")

    quality = first.get("quality_check", {})
    selected = quality.get("second_review_sample_frame_ids", [])
    minimum = math.ceil(len(frame_ids) * 0.20)
    if len(set(selected)) < minimum or not set(selected).issubset(frame_ids):
        parser.error(f"first labels must select at least {minimum} valid independent-review frame IDs")
    incomplete = [frame_id for frame_id in selected if not annotation_complete(second_tasks[frame_id].get("annotation", {}))]
    if incomplete:
        parser.error(f"independent review is incomplete for: {', '.join(incomplete)}")

    conflicts = []
    for frame_id in selected:
        a = first_tasks[frame_id].get("annotation", {})
        b = second_tasks[frame_id].get("annotation", {})
        differences = {field: {"first": a.get(field), "second": b.get(field)} for field in VISIBILITY_FIELDS if a.get(field) != b.get(field)}
        box_differences = compare_boxes(a.get("polygons", []), b.get("polygons", []), args.min_iou)
        if differences or box_differences:
            conflicts.append({"frame_id": frame_id, "visibility_differences": differences, "box_differences": box_differences})
    if conflicts:
        conflict_path = args.output.with_name(args.output.stem + ".review-conflicts.json")
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(json.dumps({"min_iou": args.min_iou, "conflicts": conflicts}, indent=2), encoding="utf-8")
        print(f"Review conflicts found in {len(conflicts)} of {len(selected)} reviewed frames. Resolve them and rerun; details: {conflict_path}")
        return 2

    for frame_id in frame_ids:
        ann = first_tasks[frame_id].get("annotation", {})
        if not annotation_complete(ann) or ann.get("boss_visibility") == "uncertain" or ann.get("event_visibility") == "uncertain":
            parser.error(f"first-pass annotation remains blank or uncertain at {frame_id}")

    frozen = dict(first)
    frozen["quality_check"] = {
        **quality,
        "draft": False,
        "frozen": True,
        "independent_review_complete": True,
        "second_review_frame_ids": sorted(set(selected)),
        "second_review_fraction": len(set(selected)) / len(frame_ids),
        "min_review_box_iou": args.min_iou,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    record = {
        "frozen_labels": str(args.output.resolve()),
        "frozen_labels_sha256": sha256(args.output.resolve()),
        "first_labels_sha256": sha256(first_path),
        "independent_review_sha256": sha256(second_path),
        "frame_manifest_sha256": sha256(manifest_path),
        "reviewed_frames": sorted(set(selected)),
        "review_fraction": len(set(selected)) / len(frame_ids),
        "review_box_iou_threshold": args.min_iou,
        "frozen_at_utc": frozen["quality_check"]["frozen_at_utc"],
    }
    args.output.with_suffix(".freeze-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
