# Optional Gemini notebook prompt

Use Gemini only to review or explain the training code. It is not the detector and must not generate ground-truth labels for frames that will be used to score the model.

Suggested prompt:

> Review this YOLOv8n training notebook for correctness. The local dataset contains reviewed bounding boxes and is split by source clip for the primary evaluation. Do not change labels, thresholds, or the held-out test set after seeing test results. Do not add upload or telemetry code. Explain each proposed code change and preserve local-only data handling.

Do not paste unredacted frames, chat, account/player names, credentials, or private project source into Gemini.
