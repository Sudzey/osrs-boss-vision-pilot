#!/usr/bin/env python3
"""Train and evaluate the YOLOv8n detection baseline locally."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Generated local data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, required=True, help="Private Runtime Data output directory")
    parser.add_argument("--name", default="yolov8n")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1 or args.imgsz < 64:
        parser.error("epochs, batch, and image size must be positive")
    args.project.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.project.resolve() / "ultralytics-config"))

    from ultralytics import YOLO

    model = YOLO(args.model)
    training = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=False,
        save=True,
        verbose=True,
    )
    best = Path(training.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Expected trained checkpoint at {best}")

    final_model = YOLO(str(best))
    metrics = final_model.val(data=str(args.data.resolve()), split="test", imgsz=args.imgsz, device=args.device, plots=False)
    summary = {
        "model": args.model,
        "best_checkpoint": str(best),
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "test_metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    Path(training.save_dir, "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.export_onnx:
        exported = final_model.export(format="onnx", imgsz=args.imgsz, simplify=False)
        print(f"ONNX export: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
