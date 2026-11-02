#!/usr/bin/env python3
"""Run the COCO-pretrained YOLOv8n model over a local frame packet."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def synchronize(device: str) -> None:
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="Local folder containing frame_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Private Runtime Data JSON output")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 Nano checkpoint; keep weights in Runtime Data")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional frame limit; 0 means all frames")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1 or args.imgsz < 64 or args.limit < 0:
        parser.error("threshold must be 0–1, imgsz at least 64, and limit nonnegative")
    manifest_path = args.packet / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"][: args.limit] if args.limit else manifest["tasks"]
    if not tasks:
        parser.error("the frame manifest contains no selected tasks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output.parent / "ultralytics-config"))

    from ultralytics import YOLO

    model = YOLO(args.model)
    first_image = args.packet / tasks[0]["image"]
    model.predict(str(first_image), imgsz=args.imgsz, conf=args.threshold, device=args.device, verbose=False, save=False)

    rows = []
    infer_ms: list[float] = []
    total_ms: list[float] = []
    for task in tasks:
        image_path = args.packet / task["image"]
        synchronize(args.device)
        started = time.perf_counter()
        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.threshold,
            device=args.device,
            verbose=False,
            save=False,
        )[0]
        synchronize(args.device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        speed = {key: round(float(value), 3) for key, value in result.speed.items()}
        infer_ms.append(speed.get("inference", 0.0))
        total_ms.append(elapsed_ms)
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": model.names[class_id],
                "score": round(float(box.conf[0]), 5),
                "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            })
        rows.append({
            "frame_id": task["frame_id"],
            "source_file": task["source_file"],
            "source_seconds": task["source_seconds"],
            "detections": detections,
            "speed_ms": speed,
            "elapsed_ms_including_predict_call": round(elapsed_ms, 3),
        })

    counts = {}
    for row in rows:
        for detection in row["detections"]:
            counts[detection["class"]] = counts.get(detection["class"], 0) + 1
    payload = {
        "run_kind": "zero-shot COCO model; no OSRS fine-tuning",
        "model_id": args.model,
        "ultralytics_version": __import__("ultralytics").__version__,
        "threshold": args.threshold,
        "imgsz": args.imgsz,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else "CPU",
        "frames": len(rows),
        "detections_by_class": dict(sorted(counts.items())),
        "inference_ms": {
            "median": round(statistics.median(infer_ms), 3) if infer_ms else None,
            "p95": round(percentile(infer_ms, 0.95), 3) if infer_ms else None,
        },
        "elapsed_ms_including_predict_call": {
            "median": round(statistics.median(total_ms), 3) if total_ms else None,
            "p95": round(percentile(total_ms, 0.95), 3) if total_ms else None,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
