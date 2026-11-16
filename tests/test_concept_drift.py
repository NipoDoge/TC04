import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.artifacts import sha256
from src.config import CONCEPT_PERIODS, TARGET
from src.contract import FEATURE_COLUMNS, schema
from src.concept_drift import build_concept_scenarios, classification_metrics, run_experiment


class FixedModel:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def predict_proba(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        scores = (features["MonthlyIncome"].to_numpy() / 10000).clip(0.01, 0.99)
        return np.column_stack([1 - scores, scores])


class ConceptDriftTests(unittest.TestCase):
    def setUp(self):
        n = 400
        self.data = pd.DataFrame({
            TARGET: np.tile([0, 1], n // 2),
            **{c: np.arange(n, dtype=float) % 5 for c in FEATURE_COLUMNS},
        })
        for name, definition in schema.columns.items():
            if str(definition.dtype) == "int64":
                self.data[name] = self.data[name].astype(int)
        self.data["age"] = 20 + np.arange(n) % 60
        self.data["MonthlyIncome"] = np.linspace(1000, 9000, n)

    def test_reproducible_conditional_nested_changes_and_fixed_features(self):
        original = self.data.copy(deep=True)
        cutoff = float(self.data.MonthlyIncome.quantile(0.25))
        scenarios, audit = build_concept_scenarios(self.data, cutoff, seed=42)
        repeated, repeated_audit = build_concept_scenarios(self.data, cutoff, seed=42)
        pd.testing.assert_frame_equal(audit, repeated_audit)
        pd.testing.assert_frame_equal(scenarios["c0_control"], self.data)
        previous = set()
        for name, _ in CONCEPT_PERIODS:
            current = scenarios[name]
            pd.testing.assert_frame_equal(current, repeated[name])
            pd.testing.assert_frame_equal(current.loc[:, FEATURE_COLUMNS], self.data.loc[:, FEATURE_COLUMNS])
            changed = current[TARGET].ne(self.data[TARGET])
            self.assertTrue(self.data.loc[changed, "MonthlyIncome"].le(cutoff).all())
            self.assertTrue(self.data.loc[changed, TARGET].eq(0).all())
            self.assertTrue(current.loc[changed, TARGET].eq(1).all())
            now = set(np.flatnonzero(changed))
            self.assertTrue(previous <= now)
            previous = now
        self.assertTrue(previous)
        other, _ = build_concept_scenarios(self.data, cutoff, seed=43)
        self.assertFalse(other["c3"][TARGET].equals(scenarios["c3"][TARGET]))
        pd.testing.assert_frame_equal(original, self.data)

    def test_invalid_cutoffs_and_single_class_are_rejected(self):
        for cutoff in [-1, np.nan, 0, 1e9]:
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                build_concept_scenarios(self.data, cutoff)
        with self.assertRaisesRegex(ValueError, "duas classes"):
            build_concept_scenarios(self.data.assign(**{TARGET: 0}), 3000)

    def test_metrics_match_known_confusion_matrix_and_handle_single_class(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]),
                                         np.array([0.1, 0.8, 0.4, 0.9]), np.array([0, 1, 0, 1]))
        self.assertEqual([metrics[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(metrics["auc"], 0.75)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["brier"], 0.255)
        # Campos também verificados por PerformanceTests na entrega final (4f50a83).
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["prevalence"], 0.5)
        self.assertEqual(metrics["predicted_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_score"], 0.55)
        self.assertEqual(metrics["ks_classification"], 0.5)
        single = classification_metrics(np.zeros(4, dtype=int), np.full(4, 0.2), np.zeros(4, dtype=int))
        self.assertIsNone(single["auc"])
        self.assertIsNone(single["recall"])
        self.assertIsNone(single["average_precision"])
        self.assertIsNone(single["ks_classification"])
        self.assertEqual(single["prevalence"], 0.0)
        self.assertEqual(single["predicted_positive_rate"], 0.0)
        self.assertAlmostEqual(single["mean_score"], 0.2)

    def write_sources(self, root):
        reference, holdout, model = root / "reference.parquet", root / "holdout.parquet", root / "model.joblib"
        self.data.to_parquet(reference)
        self.data.to_parquet(holdout)
        joblib.dump(FixedModel(), model)
        return reference, holdout, model

    def test_complete_experiment_native_reports_hashes_and_frozen_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, holdout, model = self.write_sources(root)
            before = {p: sha256(p) for p in [reference, holdout, model]}
            output = root / "experiment"
            manifest = run_experiment(reference, holdout, model, output_dir=output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
            self.assertTrue(all(manifest["invariants"].values()))
            self.assertEqual(len(pd.read_csv(output / "sensitivity.csv")), 20)
            for artifact in manifest["artifacts"]:
                self.assertEqual(sha256(output / artifact["file"]), artifact["sha256"])
            control = pd.read_parquet(output / "predictions/c0_control.parquet")
            for row in manifest["results"]:
                name = row["scenario"]
                predictions = pd.read_parquet(output / f"predictions/{name}.parquet")
                np.testing.assert_array_equal(predictions.probability_1, control.probability_1)
                np.testing.assert_array_equal(predictions.predicted_class, control.predicted_class)
                native = json.loads((output / f"classification_{name}.json").read_text())
                metrics = {m["config"]["type"].split(":")[-1]: m["value"] for m in native["metrics"]}
                self.assertAlmostEqual(metrics["RocAuc"], row["auc"])
                self.assertAlmostEqual(metrics["Recall"], row["recall"])
                self.assertEqual(sum(row[k] for k in ["tn", "fp", "fn", "tp"]), len(self.data))
            self.assertFalse(pd.read_csv(output / "feature_drift.csv").drift_detected.any())
            self.assertEqual(before, {p: sha256(p) for p in before})
            with self.assertRaises(FileExistsError):
                run_experiment(reference, holdout, model, output_dir=output)

    def test_report_failure_does_not_publish_partial_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, holdout, model = self.write_sources(root)
            with patch("src.concept_drift.performance_report", side_effect=RuntimeError("report failed")):
                with self.assertRaisesRegex(RuntimeError, "report failed"):
                    run_experiment(reference, holdout, model, output_dir=root / "output")
            self.assertFalse((root / "output").exists())
            self.assertFalse(list(root.glob(".concept-*")))


if __name__ == "__main__":
    unittest.main()
