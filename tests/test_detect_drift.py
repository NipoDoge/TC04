import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.artifacts import sha256
from src.config import SCENARIOS, TARGET
from src.contract import FEATURE_COLUMNS, schema
from src.detect_drift import add_effect_sizes, compare_datasets, detect_drift, effect_level, summarize


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
        current["MonthlyIncome"] *= 0.3
        current["DebtRatio"] *= 4
        original_reference, original_current = self.reference.copy(), current.copy()
        rows = summarize(compare_datasets(self.reference, current), "shift")
        self.assertEqual({r["feature"] for r in rows if r["drift_detected"]},
                         {"MonthlyIncome", "DebtRatio"})
        for row in rows:
            self.assertEqual(row["score_kind"], "p_value" if row["method"] == "ks" else "psi")
        pd.testing.assert_frame_equal(self.reference, original_reference)
        pd.testing.assert_frame_equal(current, original_current)

    def test_effect_size_tells_how_large_the_shift_is(self):
        rows = add_effect_sizes(summarize(compare_datasets(self.reference, self.reference), "control"),
                                self.reference, self.reference)
        self.assertTrue(all(r["effect_psi"] == 0 and r["ks_statistic"] == 0 and r["effect_level"] == "pequeno" for r in rows))
        sizes = {}
        for factor in (0.98, 0.7, 0.3):                              # mudança mínima, moderada e forte na renda
            current = self.reference.copy(deep=True)
            current["MonthlyIncome"] *= factor
            evaluation = compare_datasets(self.reference, current)
            detected = {r["feature"]: r for r in add_effect_sizes(summarize(evaluation, "shift"), self.reference, current)}
            sizes[factor] = detected["MonthlyIncome"]
            self.assertEqual(detected["DebtRatio"]["effect_psi"], 0)
            self.assertEqual(set(detected["MonthlyIncome"]) - {"effect_psi", "effect_level", "ks_statistic"},
                             {"scenario", "feature", "method", "score", "score_kind", "threshold", "drift_detected"})
        self.assertLess(sizes[0.98]["effect_psi"], sizes[0.7]["effect_psi"])
        self.assertLess(sizes[0.7]["effect_psi"], sizes[0.3]["effect_psi"])
        self.assertLess(sizes[0.98]["ks_statistic"], sizes[0.3]["ks_statistic"])
        self.assertEqual(sizes[0.98]["effect_level"], "pequeno")
        self.assertEqual(sizes[0.3]["effect_level"], "grande")
        self.assertEqual([effect_level(v) for v in (0.0, 0.0999, 0.10, 0.1999, 0.20, 3.0)],
                         ["pequeno", "pequeno", "moderado", "moderado", "grande", "grande"])

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
            self.assertTrue(summary.effect_psi.eq(0).all())           # cenários idênticos à referência
            self.assertEqual(set(summary.effect_level), {"pequeno"})
            self.assertIn("effect_size", manifest["settings"])
            self.assertEqual(manifest["scenarios"][0]["largest_effect"]["effect_psi"], 0)
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
