from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dataset import CLASSES, bbox_from_polygon, load_samples, split_grouped, split_random


def polygon(cls: str = "boss") -> dict:
    return {"class": cls, "points": [[10, 10], [30, 10], [30, 40], [10, 40]]}


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root / "packet"
        self.frames = self.packet / "frames"
        self.frames.mkdir(parents=True)
        self.labels_path = self.root / "labels.json"
        self.tasks = []
        for i in range(9):
            frame_name = f"frame_{i:03}.jpg"
            (self.frames / frame_name).write_bytes(b"fixture")
            self.tasks.append({
                "frame_id": f"GT-{i:03}",
                "boss": "Zulrah",
                "image": f"frames/{frame_name}",
                "source_file": f"clip-{i // 3}.mp4",
                "source_seconds": float(i),
                "width": 100,
                "height": 80,
                "annotation": {
                    "boss_visibility": "visible",
                    "event_visibility": "not_visible",
                    "source_markup": "none",
                    "polygons": [polygon()],
                },
            })
        self._save()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _save(self) -> None:
        payload = {
            "quality_check": {
                "draft": False,
                "frozen": True,
                "independent_review_complete": True,
                "second_review_frame_ids": ["GT-000", "GT-001"],
                "second_review_fraction": 2 / len(self.tasks),
            },
            "tasks": self.tasks,
        }
        self.labels_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_polygon_becomes_normalized_yolo_bbox_and_clips_to_image(self) -> None:
        self.assertEqual(bbox_from_polygon([[-5, 10], [30, 10], [30, 50], [-5, 50]], 100, 80), (0.15, 0.375, 0.3, 0.5))

    def test_boss_identity_is_a_distinct_detector_class(self) -> None:
        samples = load_samples(self.packet, self.labels_path)
        self.assertEqual(CLASSES[samples[0]["rows"][0][0]], "boss_zulrah")

    def test_grouped_split_keeps_source_clips_disjoint_and_all_samples(self) -> None:
        samples = load_samples(self.packet, self.labels_path)
        splits = split_grouped(samples, 7)
        group_sets = [{item["source_file"] for item in group} for group in splits.values()]
        self.assertEqual(len(set.union(*group_sets)), 3)
        self.assertEqual(sum(len(group) for group in splits.values()), 9)
        self.assertEqual(sum(bool(group) for group in splits.values()), 3)
        self.assertEqual(sum(len(samples_) for samples_ in splits.values()), len(samples))
        self.assertEqual(len(set.intersection(*group_sets)), 0)

    def test_random_split_is_deterministic_and_complete(self) -> None:
        samples = load_samples(self.packet, self.labels_path)
        first = split_random(samples, 21)
        second = split_random(samples, 21)
        self.assertEqual([[item["frame_id"] for item in first[name]] for name in first], [[item["frame_id"] for item in second[name]] for name in second])
        self.assertEqual(sum(len(group) for group in first.values()), 9)

    def test_uncertain_labels_fail_instead_of_being_silently_dropped(self) -> None:
        self.tasks[0]["annotation"]["event_visibility"] = "uncertain"
        self._save()
        with self.assertRaisesRegex(ValueError, "still has an uncertain label"):
            load_samples(self.packet, self.labels_path)

    def test_visibility_must_match_polygon_labels(self) -> None:
        self.tasks[0]["annotation"]["boss_visibility"] = "not_visible"
        self._save()
        with self.assertRaisesRegex(ValueError, "boss polygon marked not visible"):
            load_samples(self.packet, self.labels_path)

    def test_image_path_cannot_escape_packet(self) -> None:
        self.tasks[0]["image"] = "../../outside.jpg"
        self._save()
        with self.assertRaisesRegex(ValueError, "escapes the packet"):
            load_samples(self.packet, self.labels_path)

    def test_draft_labels_are_rejected(self) -> None:
        payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        payload["quality_check"]["draft"] = True
        self.labels_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed and frozen"):
            load_samples(self.packet, self.labels_path)


if __name__ == "__main__":
    unittest.main()
