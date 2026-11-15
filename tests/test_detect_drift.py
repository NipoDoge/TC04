import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.artifacts import sha256
from src.config import LATE90_COLUMN, SCENARIOS, TARGET, UTILIZATION_COLUMN
from src.contract import FEATURE_COLUMNS, schema
from src.detect_drift import compare_datasets, detect_drift, summarize


class DetectDriftTests(unittest.TestCase):
    def setUp(self):
        n = 300
        self.reference = pd.DataFrame({
            TARGET: np.tile([0, 1], n // 2),
            **{c: np.arange(n, dtype=float) % 5 for c in FEATURE_COLUMNS},
        })
        for name, definition in schema.columns.items():
            if str(definition.dtype) == "int64":
                self.reference[name] = self.reference[name].astype(int)
        self.reference["age"] = 20 + np.arange(n) % 60
        self.reference["MonthlyIncome"] = np.linspace(1000, 4000, n)
        self.reference["DebtRatio"] = np.linspace(0.1, 0.5, n)

    def test_identical_features_ignore_target_and_input_column_order(self):
        current = self.reference.copy(deep=True)
        current[TARGET] = 1 - current[TARGET]
        current = current.loc[:, current.columns[::-1]]
        evaluation = compare_datasets(self.reference, current)
        rows = summarize(evaluation, "control")
        self.assertEqual({r["feature"] for r in rows}, set(FEATURE_COLUMNS))
        self.assertFalse(any(r["drift_detected"] for r in rows))
        self.assertTrue(evaluation.dict()["tests"])

    def test_known_shifts_detected_without_modifying_inputs(self):
        current = self.reference.copy(deep=True)
        current[UTILIZATION_COLUMN] *= 4
        current[LATE90_COLUMN] += 1
        original_reference, original_current = self.reference.copy(), current.copy()
        rows = summarize(compare_datasets(self.reference, current), "shift")
        self.assertEqual({r["feature"] for r in rows if r["drift_detected"]},
                         {UTILIZATION_COLUMN, LATE90_COLUMN})
        for row in rows:
            self.assertEqual(row["score_kind"], "p_value" if row["method"] == "ks" else "psi")
        pd.testing.assert_frame_equal(self.reference, original_reference)
        pd.testing.assert_frame_equal(current, original_current)

    def test_report_publication_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, scenarios, output = root / "reference.parquet", root / "scenarios", root / "reports"
            self.reference.to_parquet(reference)
            scenarios.mkdir()
            for name in SCENARIOS:
                self.reference.to_parquet(scenarios / f"{name}.parquet")
            sources = [reference, *scenarios.glob("*.parquet")]
            before = {p: sha256(p) for p in sources}
            manifest = detect_drift(reference, scenarios, output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(len(summary), 4 * len(FEATURE_COLUMNS))
            self.assertFalse(summary.drift_detected.any())
            self.assertEqual(sha256(output / manifest["summary"]["file"]), manifest["summary"]["sha256"])
            self.assertEqual(manifest["reference"]["sha256"], before[reference])
            for scenario in manifest["scenarios"]:
                self.assertFalse(scenario["dataset_drift_detected"])
                for report in scenario["reports"]:
                    path = output / report["file"]
                    self.assertEqual(sha256(path), report["sha256"])
                    self.assertGreater(path.stat().st_size, 100)
                raw = json.loads((output / f"{scenario['name']}.json").read_text())
                self.assertEqual(len(raw["metrics"]), len(FEATURE_COLUMNS) + 1)
            self.assertEqual(before, {p: sha256(p) for p in sources})
            with self.assertRaises(FileExistsError):
                detect_drift(reference, scenarios, output)

    def test_invalid_last_scenario_does_not_publish_partial_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, output = root / "reference.parquet", root / "reports"
            self.reference.to_parquet(reference)
            for name in SCENARIOS:
                data = self.reference.copy()
                if name == SCENARIOS[-1]:
                    data.loc[0, "MonthlyIncome"] = -1
                data.to_parquet(root / f"{name}.parquet")
            with self.assertRaises(pa.errors.SchemaErrors):
                detect_drift(reference, root, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".drift-*")))


if __name__ == "__main__":
    unittest.main()
