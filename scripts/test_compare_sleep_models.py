"""Protocol checks; synthetic values test plumbing, never scientific accuracy."""
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_sleep_models import (BASE_FEATURES, temporal_features, make_target_splits,
                                  filter_probabilities, transitions, run)


def fixture():
    rng = np.random.default_rng(123)
    rows = []
    for subject in range(4):
        for night in range(1, 5):
            for epoch in range(40):
                y = (epoch // 8 + subject) % 5
                row = {f: float(rng.normal(y, 0.5)) for f in BASE_FEATURES}
                row.update(subject=f"s{subject}", night=night, epoch=epoch, sleep_stage=y)
                row["HR"] += 60
                rows.append(row)
    return pd.DataFrame(rows)


class ProtocolTests(unittest.TestCase):
    def test_future_values_and_other_nights_do_not_change_features(self):
        raw = fixture()
        a, features = temporal_features(raw)
        raw.loc[(raw.subject == "s0") & (raw.night == 1) & (raw.epoch >= 20), "HR"] = 999
        raw.loc[raw.night == 2, "HR"] = 999
        b, _ = temporal_features(raw)
        mask = (a.subject == "s0") & (a.night == 1) & (a.epoch < 20)
        pd.testing.assert_frame_equal(a.loc[mask, features], b.loc[mask, features])

    def test_fixed_target_and_nested_history(self):
        data, _ = temporal_features(fixture())
        one, m1 = make_target_splits(data, "s0", 1)
        two, m2 = make_target_splits(data, "s0", 2)
        self.assertEqual(m1["train_nights"], [2])
        self.assertEqual(m2["train_nights"], [1, 2])
        pd.testing.assert_frame_equal(one[1], two[1])
        pd.testing.assert_frame_equal(one[2], two[2])
        self.assertLess(two[0].night.max(), two[1].night.min())
        self.assertLess(two[1].night.max(), two[2].night.min())

    def test_walk_forward_splits_advance_with_history(self):
        data, _ = temporal_features(fixture())
        splits = {k: make_target_splits(data, "s0", k, mode="walk_forward")[1]
                  for k in range(3)}
        self.assertEqual(splits[0]["train_nights"], [])
        self.assertEqual(splits[0]["validation_night"], 1)
        self.assertEqual(splits[0]["test_night"], 2)
        self.assertEqual(splits[1]["train_nights"], [1])
        self.assertEqual(splits[1]["validation_night"], 2)
        self.assertEqual(splits[1]["test_night"], 3)
        self.assertEqual(splits[2]["train_nights"], [1, 2])
        self.assertEqual(splits[2]["validation_night"], 3)
        self.assertEqual(splits[2]["test_night"], 4)
        for k, meta in splits.items():
            self.assertEqual(meta["mode"], "walk_forward")
            self.assertNotIn(meta["validation_night"], meta["train_nights"])
            self.assertNotIn(meta["test_night"], meta["train_nights"])
            self.assertLess(meta["validation_night"], meta["test_night"])

    def test_unknown_labels_do_not_compress_temporal_windows(self):
        raw = fixture()
        raw.loc[(raw.subject == "s0") & (raw.night == 1) & (raw.epoch == 10), "sleep_stage"] = 5
        data, _ = temporal_features(raw)
        self.assertNotIn(5, data.sleep_stage.unique())
        self.assertIn(11, data[(data.subject == "s0") & (data.night == 1)].epoch.tolist())

    def test_filter_is_causal_and_resets(self):
        data, _ = temporal_features(fixture())
        frame = data.iloc[:80]
        p = np.full((80, 5), 0.2)
        p[:40, 0] = 0.9
        p /= p.sum(axis=1, keepdims=True)
        transition = transitions(data)
        output = filter_probabilities(p, frame, transition, 1)
        np.testing.assert_allclose(output[40], p[40])
        p[60:] = [1, 0, 0, 0, 0]
        changed = filter_probabilities(p, frame, transition, 1)
        np.testing.assert_allclose(output[:60], changed[:60])

    def test_end_to_end_matched_predictions(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            root = Path(directory)
            fixture().to_csv(root / "input.csv", index=False)
            run(SimpleNamespace(input=root / "input.csv", output=root / "result", folds=2,
                                trees=4, epochs=2, adapt_epochs=1, jobs=1, seed=42,
                                mode="walk_forward"))
            prediction = pd.read_csv(root / "result/test_predictions.csv")
            keys = ["subject", "night", "epoch", "history", "sleep_stage"]
            left = prediction[prediction.model == "random_forest"][keys].reset_index(drop=True)
            right = prediction[prediction.model == "neural_network"][keys].reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)
            manifest = json.loads((root / "result/split_manifest.json").read_text())
            for split in manifest:
                target = set(split["target_subjects"])
                source_train = {tuple(pair) for pair in split["source_train"]}
                source_val = {tuple(pair) for pair in split["source_validation"]}
                self.assertFalse(source_train & source_val)
                self.assertFalse(target & {s for s, _ in source_train | source_val})
                self.assertNotIn(split["validation_night"], split["train_nights"])
                self.assertNotIn(split["test_night"], split["train_nights"])
                self.assertLess(split["validation_night"], split["test_night"])
            metric = pd.read_csv(root / "result/metrics.csv")
            zero = metric[(metric.history == 0) & (metric.split == "test")]
            self.assertTrue((zero.alpha == 0).all())
            self.assertFalse(prediction.duplicated(["model", "subject", "night", "epoch", "history"]).any())
            self.assertTrue((root / "result/SUMMARY.md").exists())

    def test_target_test_labels_cannot_change_predictions_or_selection(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            root = Path(directory)
            raw = fixture()
            predictions, settings = [], []
            for trial in range(2):
                if trial:
                    mask = (raw.subject == "s0") & (raw.night == 4)
                    raw.loc[mask, "sleep_stage"] = (raw.loc[mask, "sleep_stage"] + 1) % 5
                raw.to_csv(root / "input.csv", index=False)
                output = root / f"trial{trial}"
                run(SimpleNamespace(input=root / "input.csv", output=output, folds=2,
                                    trees=4, epochs=2, adapt_epochs=1, jobs=1, seed=42,
                                    mode="anchored"))
                pred = pd.read_csv(output / "test_predictions.csv")
                predictions.append(pred[pred.subject == "s0"].drop(columns="sleep_stage").reset_index(drop=True))
                metric = pd.read_csv(output / "metrics.csv")
                settings.append(metric[metric.subject == "s0"][["history", "model", "split", "alpha", "smoothing"]].reset_index(drop=True))
            pd.testing.assert_frame_equal(predictions[0], predictions[1])
            pd.testing.assert_frame_equal(settings[0], settings[1])


if __name__ == "__main__":
    unittest.main()
