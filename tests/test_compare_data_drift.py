import json
import os
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.artifacts import sha256, snapshot
from src.compare_data_drift import compare_data_drift
from src.config import SCENARIOS, TARGET
from src.contract import FEATURE_COLUMNS, schema
from src.detect_drift import detect_drift
from src.predict import predict
from src.simulate_drift import simulate


class CompareDataDriftTests(unittest.TestCase):
    def test_end_to_end_comparison_and_stale_artifact_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp)
                n = 200
                data = pd.DataFrame({TARGET: (np.arange(n) >= n // 2).astype(int),
                                     **{c: np.arange(n, dtype=float) % 5 for c in FEATURE_COLUMNS}})
                for name, definition in schema.columns.items():
                    if str(definition.dtype) == "int64":
                        data[name] = data[name].astype(int)
                data["age"] = 20 + np.arange(n) % 60
                data["MonthlyIncome"] = np.linspace(1000, 9000, n)
                for path in [Path("data/reference/reference.parquet"), Path("data/production/holdout.parquet")]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(path)
                model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, n_jobs=1)
                model.fit(data.loc[:, FEATURE_COLUMNS], data[TARGET])
                Path("models/current").mkdir(parents=True)
                joblib.dump(model, "models/current/model.joblib")

                archive = Path("artifacts/data_drift_income_debt")
                (archive / "scenarios").mkdir(parents=True)
                periods = []
                for name, income_factor, debt_factor in zip(SCENARIOS, [1, .9, .75, .6], [1, 1.1, 1.3, 1.6]):
                    changed = data.copy()
                    changed["MonthlyIncome"] *= income_factor
                    changed["DebtRatio"] *= debt_factor
                    path = archive / "scenarios" / f"{name}.parquet"
                    changed.to_parquet(path)
                    periods.append({"name": name, "file": path.name, "sha256": sha256(path)})
                (archive / "scenarios/manifest.json").write_text(json.dumps({
                    "source": snapshot("data/production/holdout.parquet")[1], "periods": periods,
                }))
                predict(scenarios_dir=archive / "scenarios", output_dir=archive / "predictions")
                detect_drift(scenarios_dir=archive / "scenarios", output_dir=archive / "reports")
                files = [{"archived_file": str(p.relative_to(archive)), "sha256": sha256(p)}
                         for p in archive.rglob("*") if p.is_file()]
                (archive / "archive.json").write_text(json.dumps({"files": files}))

                simulate()
                predict()
                detect_drift()
                protected = {p: sha256(p) for directory in ["data", "models", "artifacts", "reports/drift"]
                             for p in Path(directory).rglob("*") if p.is_file()}
                metrics = compare_data_drift()
                output = Path("reports/drift_comparison")
                self.assertEqual(len(metrics), 8)
                controls = metrics[metrics.scenario == "t0_control"]
                self.assertTrue(controls.changed_predictions.eq(0).all())
                self.assertTrue(controls.auc_drop.eq(0).all())
                self.assertEqual(len(pd.read_csv(output / "feature_importance.csv")), len(FEATURE_COLUMNS))
                self.assertEqual(len(pd.read_csv(output / "joint_importance.csv")), 2)
                self.assertEqual(len(pd.read_csv(output / "drift_summary.csv")), 80)
                manifest = json.loads((output / "manifest.json").read_text())
                for artifact in manifest["artifacts"]:
                    self.assertEqual(sha256(output / artifact["file"]), artifact["sha256"])
                self.assertEqual(protected, {p: sha256(p) for p in protected})
                with self.assertRaises(FileExistsError):
                    compare_data_drift()

                prediction = Path("data/predictions/t1.parquet")
                before = prediction.read_bytes()
                prediction.write_bytes(b"invalid prediction snapshot")
                with self.assertRaisesRegex(ValueError, "Hash de predições"):
                    compare_data_drift(output_dir="reports/invalid")
                self.assertFalse(Path("reports/invalid").exists())
                prediction.write_bytes(before)
                report_path = Path("reports/drift/manifest.json")
                report_before = report_path.read_bytes()
                report = json.loads(report_before)
                report["scenarios"][1]["source"]["sha256"] = "0" * 64
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "não corresponde ao cenário"):
                    compare_data_drift(output_dir="reports/stale_drift")
                self.assertFalse(Path("reports/stale_drift").exists())
                report_path.write_bytes(report_before)
                report = json.loads(report_before)
                report["reference"]["sha256"] = "0" * 64
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "mesma referência"):
                    compare_data_drift(output_dir="reports/other_reference")
                self.assertFalse(Path("reports/other_reference").exists())
                report_path.write_bytes(report_before)
                old_snapshot = archive / "scenarios/t1.parquet"
                old_snapshot.write_bytes(b"invalid historical snapshot")
                with self.assertRaisesRegex(ValueError, "snapshot preservado"):
                    compare_data_drift(output_dir="reports/invalid_archive")
                self.assertFalse(Path("reports/invalid_archive").exists())
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
