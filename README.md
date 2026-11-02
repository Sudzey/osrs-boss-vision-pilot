# OSRS vision detector pilot

An isolated, reproducible experiment for comparing YOLOv8n detection with the project's existing boss and event recognition methods. This repository is not integrated into Razer Inputs and contains no game footage, labels, trained weights, credentials, or runtime data.

## What it measures

- Boss and event bounding-box detection on identical, locally supplied frames.
- A Botting Hub-style random 70/20/10 frame split and a source-clip-grouped split. The grouped split is the meaningful generalization check because adjacent frames can otherwise leak across splits.
- Precision, recall, mAP50, mAP50:95, and per-frame latency from Ultralytics validation.
- ONNX export as a separate deployment-format check; export is not an accuracy result.

The pilot detector separates six boss identities (`boss_zulrah`, `boss_vorkath`, `boss_great_olm`, `boss_verzik`, `boss_maggot_king`, `boss_yama`) and generic `animation`, `projectile`, and `other` boxes. The current labels do not consistently identify each named attack, so attack-specific classification is not claimed. Keep labels and frames in the private Runtime Data directory. The preparation script points to those images in place and writes only YOLO text labels and split manifests there; it does not copy or upload images.

## Local setup

Use Python 3.10–3.14 and an NVIDIA CUDA-enabled PyTorch build for GPU training. Install dependencies in an isolated environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Prepare a dataset from the local annotation-kit export. Labels must already be reviewed; uncertain or blank frames are rejected:

```powershell
python scripts\prepare_dataset.py `
  --packet "C:\path\to\ground-truth-labeling-v1" `
  --labels "C:\path\to\Runtime Data\ground_truth_labels.json" `
  --output "C:\path\to\Runtime Data\yolo-random" `
  --split random
```

Run the clip-held-out dataset separately:

```powershell
python scripts\prepare_dataset.py `
  --packet "C:\path\to\ground-truth-labeling-v1" `
  --labels "C:\path\to\Runtime Data\ground_truth_labels.json" `
  --output "C:\path\to\Runtime Data\yolo-grouped" `
  --split grouped
```

Before labels are ready, an optional COCO-pretrained smoke run can check whether a stock YOLOv8n model recognizes any OSRS-specific targets. It does not measure boss accuracy. Run from a private Runtime Data working directory so the downloaded checkpoint and JSON stay out of this repository:

```powershell
Set-Location "C:\path\to\Runtime Data"
python "C:\path\to\detector-pilot-public\scripts\run_yolov8_zeroshot.py" `
  --packet "C:\path\to\ground-truth-labeling-v1" `
  --output ".\yolov8n-zero-shot.json"
```

Train the faithful YOLOv8 Nano baseline locally (the public guide's 100–200 epoch range is configurable):

```powershell
Set-Location "C:\path\to\Runtime Data"
python "C:\path\to\detector-pilot-public\scripts\train_yolov8.py" `
  --data ".\yolo-grouped\data.yaml" `
  --project ".\yolov8n-runs" `
  --epochs 150
```
Running from Runtime Data keeps downloaded checkpoints, Ultralytics settings/cache, and training outputs outside this public repository. Do not publish the generated `.pt` weights.

Do not publish frames, labels, weights, prediction JSON, or screenshots. To use another service such as Gemini, Roboflow, or Colab, first create a redacted copy and review that service's data terms. This code does not upload anything.

## Scope and limits

The existing packet has only 80 sampled frames across 14 source clips. Several bosses occur in just one clip, some clips contain guide markup, and a second independent label review is still needed. A random split is included only to replicate the public recipe; it can overstate performance. The clip-grouped result is primary, but it still cannot support broad full-fight accuracy claims with this packet.

YOLO returns boxes for trained categories. It does not provide SAM-style masks or temporal identity tracking by itself. Compare those separately on the same frozen frames and labels. A detector win on this pilot would justify a larger evaluation, not a product replacement.

## Licensing

This stand-alone evaluation project is released under AGPL-3.0. The Ultralytics YOLO code and trained model are also governed by Ultralytics' published AGPL-3.0 terms. Keeping this evaluation project public does not authorize copying YOLO into a private or closed-source application. The Razer Inputs app and all local data remain outside this repository.

Botting Hub's public recipe is an external claim, not independent proof of its shipped client's internals. Gemini is a code/analysis helper in that recipe, not the runtime detector.
