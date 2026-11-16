"""Adaptação de tests/test_monitor.py da entrega final (4f50a83).

Exercita os scripts existentes, com XGBoost e Evidently reais, sem introduzir
monitoramento, alertas ou dashboards ausentes nesta branch.
"""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import recall_score, roc_auc_score, roc_curve

from src.artifacts import sha256
from src.config import CONCEPT_PERIODS, DATA_DRIFT_FEATURES, SCENARIOS, TARGET, THRESHOLD
from src.concept_drift import run_experiment
from src.contract import FEATURE_COLUMNS
from src.detect_drift import detect_drift
from src.ingest import CONTRACT_REJECTED, OK, ingest
from src.make_bad_batch import make_batches
from src.predict import predict, predict_dataset
from src.prepare import prepare_datasets
from src.simulate_drift import simulate
from src.train import train
from support import synthetic_credit


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        datasets, _, _ = prepare_datasets(synthetic_credit())
        cls.inputs = {}
        for name, data in datasets.items():
            path = cls.root / f"{name}.parquet"
            data.to_parquet(path)
            cls.inputs[name] = path
        cls.datasets = datasets
        cls.models = cls.root / "models"
        cls.baseline = train(cls.inputs["train"], cls.inputs["validation"], cls.models)
        cls.run_dir = (cls.models / "current").resolve()
        cls.model_path = cls.run_dir / "model.joblib"
        cls.model = joblib.load(cls.model_path)
        cls.sources_before = {p: sha256(p) for p in [*cls.inputs.values(),
                                                   cls.model_path, cls.run_dir / "metrics.json"]}
        cls.scenarios = cls.root / "scenarios"
        cls.predictions = cls.root / "predictions"
        cls.reports = cls.root / "reports"
        cls.simulation = simulate(cls.inputs["holdout"], cls.scenarios)
        cls.inference = predict(cls.model_path, cls.inputs["train"], cls.scenarios, cls.predictions)
        cls.drift = detect_drift(cls.inputs["train"], cls.scenarios, cls.reports)
        cls.concept_dir = cls.root / "concept"
        cls.concept = run_experiment(cls.inputs["train"], cls.inputs["holdout"], cls.model_path,
                                     output_dir=cls.concept_dir)

    def test_every_scenario_is_accepted_and_accounted_for(self):
        self.assertEqual([r["name"] for r in self.simulation["periods"]], list(SCENARIOS))
        self.assertEqual([r["name"] for r in self.drift["scenarios"]], list(SCENARIOS))
        self.assertEqual([r["name"] for r in self.inference["datasets"]], ["reference", *SCENARIOS])
        for name in SCENARIOS:
            path = self.scenarios / f"{name}.parquet"
            self.assertEqual(ingest(path), OK)
            self.assertTrue(path.exists())
            self.assertEqual(len(pd.read_parquet(path)), len(self.datasets["holdout"]))

    def test_corrupted_delivery_is_quarantined_and_rejected_before_model_call(self):
        incoming = self.root / "incoming"
        make_batches(self.inputs["holdout"], incoming)
        bad_path = incoming / "bad_batch.parquet"
        bad = pd.read_parquet(bad_path)
        with patch.object(self.model, "predict_proba") as infer:
            with self.assertRaises(pa.errors.SchemaErrors):
                predict_dataset(self.model, bad)
            infer.assert_not_called()
        # A ingestão usa quarentena relativa ao cwd; confiná-la ao temporário.
        with contextlib.chdir(self.root), self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(bad_path), CONTRACT_REJECTED)
        self.assertFalse(bad_path.exists())
        quarantined = self.root / "data/quarantine/bad_batch.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(quarantined), bad)
        self.assertGreater((quarantined.parent / "bad_batch_errors.csv").stat().st_size, 0)
        self.assertEqual(ingest(incoming / "good_batch.parquet"), OK)

    def test_predictions_match_the_same_saved_model_for_every_input(self):
        for entry in self.inference["datasets"]:
            source = pd.read_parquet(entry["source"]["path"])
            saved = pd.read_parquet(self.predictions / entry["file"])
            expected = self.model.predict_proba(source.loc[:, FEATURE_COLUMNS])
            np.testing.assert_array_equal(saved[["probability_0", "probability_1"]], expected)
            np.testing.assert_array_equal(saved["predicted_class"], expected[:, 1] >= THRESHOLD)
            np.testing.assert_array_equal(saved[TARGET], source[TARGET])
            np.testing.assert_array_equal(saved["row_position"], np.arange(len(source)))

    def test_baseline_metrics_belong_to_the_published_model_and_validation(self):
        validation = self.datasets["validation"]
        scores = self.model.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1]
        fpr, tpr, _ = roc_curve(validation[TARGET], scores)
        self.assertEqual(self.baseline, json.loads((self.run_dir / "metrics.json").read_text()))
        self.assertAlmostEqual(self.baseline["auc"], roc_auc_score(validation[TARGET], scores))
        self.assertAlmostEqual(self.baseline[f"recall_at_{THRESHOLD:g}"],
                               recall_score(validation[TARGET], scores >= THRESHOLD))
        self.assertAlmostEqual(self.baseline["ks"], (tpr - fpr).max())

    def test_drift_reports_identify_the_intervened_features(self):
        shocked = next(r for r in self.drift["scenarios"] if r["name"] == "t3")
        self.assertTrue(set(DATA_DRIFT_FEATURES) <= set(shocked["drifted_features"]))
        summary = pd.read_csv(self.reports / "summary.csv")
        self.assertEqual(len(summary), len(SCENARIOS) * len(FEATURE_COLUMNS))
        for entry in self.drift["scenarios"]:
            rows = summary[summary.scenario == entry["name"]]
            self.assertEqual(set(rows.loc[rows.drift_detected, "feature"]), set(entry["drifted_features"]))
            self.assertTrue((self.reports / f"{entry['name']}.html").is_file())

    def test_data_and_concept_drift_isolate_different_mechanisms(self):
        holdout = self.datasets["holdout"]
        for name in SCENARIOS:
            current = pd.read_parquet(self.scenarios / f"{name}.parquet")
            pd.testing.assert_series_equal(current[TARGET], holdout[TARGET])
        control = pd.read_parquet(self.concept_dir / "predictions/c0_control.parquet")
        for name, _ in CONCEPT_PERIODS:
            current = pd.read_parquet(self.concept_dir / "datasets" / f"{name}.parquet")
            pd.testing.assert_frame_equal(current[FEATURE_COLUMNS], holdout[FEATURE_COLUMNS])
            predicted = pd.read_parquet(self.concept_dir / "predictions" / f"{name}.parquet")
            pd.testing.assert_frame_equal(predicted.drop(columns=TARGET), control.drop(columns=TARGET))
        changed = pd.read_parquet(self.concept_dir / "datasets/c3.parquet")
        self.assertGreater(int(changed[TARGET].sum()), int(holdout[TARGET].sum()))
        self.assertFalse(pd.read_csv(self.concept_dir / "feature_drift.csv").drift_detected.any())

    def test_sources_model_and_cross_stage_provenance_are_preserved(self):
        self.assertEqual(self.sources_before, {p: sha256(p) for p in self.sources_before})
        self.assertEqual(self.inference["model"]["sha256"], self.sources_before[self.model_path])
        self.assertEqual(self.concept["sources"]["model"], self.inference["model"])
        self.assertEqual(self.drift["reference"], self.concept["sources"]["reference"])
        self.assertEqual(self.simulation["source"], self.concept["sources"]["holdout"])
        predictions = {r["name"]: r for r in self.inference["datasets"]}
        for entry in self.drift["scenarios"]:
            self.assertEqual(entry["source"], predictions[entry["name"]]["source"])

    def test_report_failure_preserves_successful_predictions_and_existing_evidence(self):
        protected = {p: sha256(p) for p in self.root.rglob("*") if p.is_file()}
        output = self.root / "failed-reports"
        with patch("src.detect_drift.compare_datasets", side_effect=RuntimeError("Evidently indisponível")):
            with self.assertRaisesRegex(RuntimeError, "Evidently indisponível"):
                detect_drift(self.inputs["train"], self.scenarios, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".drift-*")))
        self.assertEqual(protected, {p: sha256(p) for p in protected})


if __name__ == "__main__":
    unittest.main()
