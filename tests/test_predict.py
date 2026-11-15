import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.contract import FEATURE_COLUMNS, schema
from src.artifacts import snapshot
from src.config import SCENARIOS
from src.predict import predict, predict_dataset


class RecordingModel:
    feature_names_in_ = np.array(FEATURE_COLUMNS)
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        if list(features.columns) != FEATURE_COLUMNS:
            raise AssertionError("Ordem incorreta ou vazamento do alvo")
        self.seen = features.copy()
        p = np.array([0.0, 0.49, 0.5, 1.0])
        return np.column_stack([1 - p, p])


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame({
            "SeriousDlqin2yrs": [0, 1, 0, 1],
            **{c: np.arange(4, dtype=float) + 20 for c in FEATURE_COLUMNS},
        }, index=[41, 5, 6, 99])
        for name, column in schema.columns.items():
            if str(column.dtype) == "int64":
                self.dataset[name] = self.dataset[name].astype(int)

    def test_order_target_exclusion_threshold_and_row_alignment(self):
        model = RecordingModel()
        original = self.dataset.copy(deep=True)
        result = predict_dataset(model, self.dataset.loc[:, self.dataset.columns[::-1]])
        pd.testing.assert_frame_equal(model.seen, original.loc[:, FEATURE_COLUMNS])
        self.assertEqual(result.row_position.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.SeriousDlqin2yrs.tolist(), [0, 1, 0, 1])
        self.assertEqual(result.predicted_class.tolist(), [0, 0, 1, 1])
        np.testing.assert_allclose(result.probability_0 + result.probability_1, 1)
        pd.testing.assert_frame_equal(self.dataset, original)

    def test_invalid_input_or_probabilities_are_rejected(self):
        invalid = self.dataset.drop(columns="MonthlyIncome")
        with self.assertRaises(pa.errors.SchemaErrors):
            predict_dataset(RecordingModel(), invalid)
        for probabilities in [np.full((4, 2), np.nan), np.full((4, 2), 0.8),
                              np.full((4, 1), 0.5), np.tile([-0.1, 1.1], (4, 1))]:
            with self.subTest(probabilities=probabilities):
                model = RecordingModel()
                with patch.object(model, "predict_proba", return_value=probabilities):
                    with self.assertRaises(ValueError):
                        predict_dataset(model, self.dataset)

    def test_publication_provenance_inputs_preserved_and_collision_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path, reference = root / "model.joblib", root / "reference.parquet"
            joblib.dump(RecordingModel(), model_path)
            self.dataset.to_parquet(reference)
            scenarios, output = root / "scenarios", root / "predictions"
            scenarios.mkdir()
            for name in SCENARIOS:
                self.dataset.to_parquet(scenarios / f"{name}.parquet")
            paths = [model_path, reference, *scenarios.glob("*.parquet")]
            before = {p: snapshot(p)[1] for p in paths}
            manifest = predict(model_path, reference, scenarios, output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
            self.assertEqual(manifest["model"], before[model_path])
            self.assertEqual(len(manifest["datasets"]), 5)
            for entry in manifest["datasets"]:
                self.assertIn(entry["source"], before.values())
                path = output / entry["file"]
                self.assertEqual(snapshot(path)[1]["sha256"], entry["sha256"])
                self.assertEqual(pd.read_parquet(path).predicted_class.tolist(), [0, 0, 1, 1])
            self.assertEqual(before, {p: snapshot(p)[1] for p in paths})
            with self.assertRaises(FileExistsError):
                predict(model_path, reference, scenarios, output)

    def test_missing_late_scenario_does_not_publish_partial_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path, reference = root / "model.joblib", root / "reference.parquet"
            joblib.dump(RecordingModel(), model_path)
            self.dataset.to_parquet(reference)
            for name in SCENARIOS[:-1]:
                self.dataset.to_parquet(root / f"{name}.parquet")
            with self.assertRaises(FileNotFoundError):
                predict(model_path, reference, root, root / "output")
            self.assertFalse((root / "output").exists())
            self.assertFalse(list(root.glob(".predictions-*")))


if __name__ == "__main__":
    unittest.main()
