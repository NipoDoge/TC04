import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera.pandas as pa
import joblib

from src.contract import FEATURE_COLUMNS
from src.ingest import ingest
from src.train import TrainingDataError, main as train_main, publish_artifacts, train

ROOT = Path(__file__).resolve().parents[1]


def valid_dataset():
    n = 20
    return pd.DataFrame({
        "SeriousDlqin2yrs": np.tile([0, 1], n // 2),
        "age": np.full(n, 45),
        "MonthlyIncome": np.arange(n, dtype=float) + 1000,
        "DebtRatio": np.full(n, 0.5),
        "RevolvingUtilizationOfUnsecuredLines": np.full(n, 0.3),
        "NumberOfDependents": np.zeros(n),
        **{column: np.zeros(n, dtype=int) for column in [
            "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines",
        ]},
    })


class WorkingDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.previous)

    def write_batch(self, df=None, path="data/incoming/batch.parquet"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        (valid_dataset() if df is None else df).to_parquet(path)
        return path

    def invalid_batch(self):
        df = valid_dataset()
        df.loc[0, "age"] = 15
        return self.write_batch(df)


class IngestTests(WorkingDirectoryTests):
    def test_empty_batch_is_rejected_with_quarantine_evidence(self):
        path = self.write_batch(valid_dataset().iloc[:0])
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 1)
        self.assertFalse(path.exists())
        failures = pd.read_csv("data/quarantine/batch_errors.csv")
        self.assertIn("non_empty", failures["check"].values)

    def test_infinite_numbers_are_rejected(self):
        original = valid_dataset()
        for column in original.columns:
            for value, suffix in [(np.inf, "positive"), (-np.inf, "negative")]:
                with self.subTest(column=column, value=value):
                    df = original.astype({column: float})
                    df.loc[0, column] = value
                    name = f"{column}_{suffix}"
                    path = self.write_batch(df, f"data/incoming/{name}.parquet")
                    with self.assertLogs("ingest", level="ERROR"):
                        self.assertEqual(ingest(path), 1)
                    self.assertFalse(path.exists())
                    failures = pd.read_csv(f"data/quarantine/{name}_errors.csv")
                    self.assertIn("finite", failures["check"].values)

    def test_duplicate_rows_are_rejected(self):
        df = valid_dataset()
        path = self.write_batch(pd.concat([df, df.iloc[[0]]], ignore_index=True))
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 1)
        failures = pd.read_csv("data/quarantine/batch_errors.csv")
        self.assertTrue(failures["check"].str.contains("unique", na=False).any())

    def test_missing_extra_columns_and_wrong_types_are_rejected(self):
        missing = valid_dataset().drop(columns="NumberOfDependents")
        extra = valid_dataset().assign(unexpected="value")
        wrong_type = valid_dataset().astype({"age": str})
        for name, df in [("missing_column", missing), ("extra_column", extra), ("wrong_type", wrong_type)]:
            with self.subTest(name=name):
                path = self.write_batch(df, f"data/incoming/{name}.parquet")
                with self.assertLogs("ingest", level="ERROR"):
                    self.assertEqual(ingest(path), 1)
                self.assertFalse(path.exists())
                self.assertFalse(pd.read_csv(f"data/quarantine/{name}_errors.csv").empty)

    def test_age_boundaries(self):
        for age, expected in [(17, 1), (18, 0), (100, 0), (101, 1)]:
            with self.subTest(age=age):
                df = valid_dataset()
                df.loc[0, "age"] = age
                path = self.write_batch(df, f"data/incoming/age_{age}.parquet")
                with self.assertLogs("ingest", level="INFO"):
                    self.assertEqual(ingest(path), expected)

    def test_invalid_target_null_income_and_negative_count_are_rejected(self):
        for column, value in [("SeriousDlqin2yrs", 2), ("MonthlyIncome", np.nan),
                              ("NumberOfOpenCreditLinesAndLoans", -1)]:
            with self.subTest(column=column):
                df = valid_dataset()
                df.loc[0, column] = value
                path = self.write_batch(df, f"data/incoming/{column}.parquet")
                with self.assertLogs("ingest", level="ERROR"):
                    self.assertEqual(ingest(path), 1)

    def test_missing_cli_argument_returns_two(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.ingest"],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2)

    def test_valid_batch_returns_zero_and_stays_at_source(self):
        path = self.write_batch()
        self.assertEqual(ingest(path), 0)
        self.assertTrue(path.exists())
        self.assertFalse(Path("data/quarantine").exists())

    def test_contract_rejection_returns_one_with_evidence(self):
        path = self.invalid_batch()
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 1)
        self.assertFalse(path.exists())
        self.assertTrue(Path("data/quarantine/batch.parquet").exists())
        self.assertFalse(pd.read_csv("data/quarantine/batch_errors.csv").empty)

    def test_missing_or_corrupted_input_returns_two(self):
        Path("corrupt.parquet").write_text("not a parquet file")
        for path in ["missing.parquet", "corrupt.parquet"]:
            with self.subTest(path=path), self.assertLogs("ingest", level="ERROR"):
                self.assertEqual(ingest(path), 2)
        self.assertFalse(Path("data/quarantine").exists())

    def test_quarantine_directory_failure_returns_two(self):
        path = self.invalid_batch()
        Path("data/quarantine").write_text("directory unavailable")
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 2)
        self.assertTrue(path.exists())

    def test_report_write_failure_returns_two_and_preserves_source(self):
        path = self.invalid_batch()
        with patch("pandas.DataFrame.to_csv", side_effect=PermissionError("denied")):
            with self.assertLogs("ingest", level="ERROR"):
                self.assertEqual(ingest(path), 2)
        self.assertTrue(path.exists())
        self.assertFalse(Path("data/quarantine/batch.parquet").exists())

    def test_move_failure_returns_two_and_preserves_source(self):
        path = self.invalid_batch()
        with patch("src.ingest.shutil.move", side_effect=OSError("move failed")):
            with self.assertLogs("ingest", level="ERROR"):
                self.assertEqual(ingest(path), 2)
        self.assertTrue(path.exists())

    def test_existing_quarantine_is_preserved(self):
        path = self.invalid_batch()
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 1)
        old = Path("data/quarantine/batch.parquet").read_bytes()
        self.write_batch()
        # A nova ocorrência inválida não pode substituir a evidência anterior.
        path = self.invalid_batch()
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(path), 2)
        self.assertEqual(Path("data/quarantine/batch.parquet").read_bytes(), old)
        self.assertTrue(path.exists())

    def test_cli_exit_codes(self):
        good = self.write_batch(path="good.parquet")
        bad = valid_dataset(); bad.loc[0, "age"] = 15
        self.write_batch(bad, "bad.parquet")
        self.write_batch(valid_dataset().iloc[:0], "empty.parquet")
        infinite = valid_dataset(); infinite.loc[0, "MonthlyIncome"] = np.inf
        self.write_batch(infinite, "infinite.parquet")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        for path, expected in [(str(good), 0), ("bad.parquet", 1), ("empty.parquet", 1),
                               ("infinite.parquet", 1), ("missing.parquet", 2)]:
            with self.subTest(path=path):
                result = subprocess.run(
                    [sys.executable, "-m", "src.ingest", path], env=env,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, expected, result.stderr)


class TrainingTests(WorkingDirectoryTests):
    def write_partitions(self, ref, val):
        self.write_batch(ref, "data/reference/reference.parquet")
        self.write_batch(val, "data/validation/validation.parquet")

    def test_missing_column_and_duplicates_block_training(self):
        original = valid_dataset()
        invalids = [original.drop(columns="MonthlyIncome"),
                    pd.concat([original, original.iloc[[0]]], ignore_index=True)]
        for partition in ["reference", "validation"]:
            for invalid in invalids:
                with self.subTest(partition=partition, rows=len(invalid), columns=len(invalid.columns)):
                    ref, val = valid_dataset(), valid_dataset()
                    if partition == "reference": ref = invalid
                    else: val = invalid
                    self.write_partitions(ref, val)
                    with patch("src.train.XGBClassifier") as model:
                        with self.assertRaises(pa.errors.SchemaErrors):
                            train()
                        model.assert_not_called()
                    self.assertFalse(Path("models").exists())

    def test_invalid_reference_or_validation_blocks_model_creation(self):
        for partition in ["reference", "validation"]:
            for column, value in [("age", 15), ("MonthlyIncome", np.nan), ("DebtRatio", -1.0)]:
                with self.subTest(partition=partition, column=column):
                    ref, val = valid_dataset(), valid_dataset()
                    (ref if partition == "reference" else val).loc[0, column] = value
                    self.write_partitions(ref, val)
                    with patch("src.train.XGBClassifier") as model:
                        with self.assertRaises(pa.errors.SchemaErrors):
                            train()
                        model.assert_not_called()
                    self.assertFalse(Path("models").exists())

    def test_empty_or_single_class_partition_blocks_training(self):
        for partition in ["reference", "validation"]:
            for invalid in [valid_dataset().iloc[:0], valid_dataset().query("SeriousDlqin2yrs == 0")]:
                with self.subTest(partition=partition, rows=len(invalid)):
                    ref, val = valid_dataset(), valid_dataset()
                    if partition == "reference": ref = invalid
                    else: val = invalid
                    self.write_partitions(ref, val)
                    with patch("src.train.XGBClassifier") as model:
                        expected_error = pa.errors.SchemaErrors if invalid.empty else TrainingDataError
                        with self.assertRaises(expected_error):
                            train()
                        model.assert_not_called()
                    self.assertFalse(Path("models").exists())

    def test_infinite_values_block_training_before_model_creation(self):
        for partition in ["reference", "validation"]:
            for column in ["MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]:
                for value in [np.inf, -np.inf]:
                    with self.subTest(partition=partition, column=column, value=value):
                        ref, val = valid_dataset(), valid_dataset()
                        (ref if partition == "reference" else val).loc[0, column] = value
                        self.write_partitions(ref, val)
                        with patch("src.train.XGBClassifier") as model:
                            with self.assertRaises(pa.errors.SchemaErrors):
                                train()
                            model.assert_not_called()
                        with self.assertLogs("train", level="ERROR"):
                            self.assertEqual(train_main(), 1)
                        self.assertFalse(Path("models").exists())

    def test_training_is_invariant_to_input_column_order(self):
        ref = valid_dataset()
        # Variar as entradas para que a comparação de predições seja informativa.
        ref["DebtRatio"] = np.linspace(0, 1, len(ref))
        val = ref.copy()
        val["MonthlyIncome"] += 10000
        reference = self.write_batch(ref, "reference.parquet")
        validation = self.write_batch(val, "validation.parquet")
        baseline_metrics = train(reference, validation, "baseline")
        baseline = joblib.load("baseline/current/model.joblib")
        reference = self.write_batch(ref[ref.columns[::-1]], "reference_shuffled.parquet")
        validation = self.write_batch(val.sample(frac=1, axis=1, random_state=42), "validation_shuffled.parquet")
        shuffled_metrics = train(reference, validation, "shuffled")
        shuffled = joblib.load("shuffled/current/model.joblib")
        self.assertEqual(shuffled_metrics, baseline_metrics)
        self.assertEqual(shuffled.get_booster().feature_names, FEATURE_COLUMNS)
        np.testing.assert_allclose(
            baseline.predict_proba(val.loc[:, FEATURE_COLUMNS]),
            shuffled.predict_proba(val.loc[:, FEATURE_COLUMNS]),
        )

    def test_main_distinguishes_contract_and_operational_failure(self):
        with self.assertLogs("train", level="ERROR"):
            self.assertEqual(train_main(), 2)
        ref = valid_dataset(); ref.loc[0, "age"] = 15
        self.write_partitions(ref, valid_dataset())
        Path("models").mkdir()
        previous_model = Path("models/model.joblib")
        previous_model.write_bytes(b"previous valid model")
        with self.assertLogs("train", level="ERROR"):
            self.assertEqual(train_main(), 1)
        self.assertEqual(previous_model.read_bytes(), b"previous valid model")


class ArtifactPublicationTests(WorkingDirectoryTests):
    def publish_previous(self):
        run = publish_artifacts({"version": "previous"}, {"auc": 0.75}, "models")
        snapshot = {name: (run / name).read_bytes() for name in ["model.joblib", "metrics.json"]}
        return run.resolve(), snapshot

    def assert_previous_unchanged(self, run, snapshot):
        self.assertEqual(Path("models/current").resolve(), run)
        for name, content in snapshot.items():
            self.assertEqual((Path("models/current") / name).read_bytes(), content)
        self.assertEqual(list(Path("models/runs").iterdir()), [run.relative_to(Path.cwd())])
        self.assertFalse(list(Path("models").glob(".tmp-*")))
        self.assertFalse(list(Path("models").glob(".current-*")))

    def test_success_publishes_complete_version_and_keeps_previous(self):
        old, snapshot = self.publish_previous()
        new = publish_artifacts({"version": "new"}, {"auc": 0.85}, "models")
        self.assertEqual(Path("models/current").resolve(), new.resolve())
        self.assertEqual(joblib.load("models/current/model.joblib"), {"version": "new"})
        self.assertEqual(json.loads(Path("models/current/metrics.json").read_text()), {"auc": 0.85})
        for name, content in snapshot.items():
            self.assertEqual((old / name).read_bytes(), content)
        self.assertFalse(list(Path("models").glob(".tmp-*")))

    def test_metrics_write_failure_preserves_previous_set(self):
        old, snapshot = self.publish_previous()
        with patch("pathlib.Path.write_text", side_effect=OSError("metrics write failed")):
            with self.assertRaises(OSError):
                publish_artifacts({"version": "new"}, {"auc": 0.85}, "models")
        self.assert_previous_unchanged(old, snapshot)

    def test_model_write_failure_preserves_previous_set(self):
        old, snapshot = self.publish_previous()
        with patch("src.train.joblib.dump", side_effect=OSError("model write failed")):
            with self.assertRaises(OSError):
                publish_artifacts({"version": "new"}, {"auc": 0.85}, "models")
        self.assert_previous_unchanged(old, snapshot)

    def test_publication_failure_preserves_previous_set(self):
        old, snapshot = self.publish_previous()
        with patch("src.train.os.replace", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                publish_artifacts({"version": "new"}, {"auc": 0.85}, "models")
        self.assert_previous_unchanged(old, snapshot)

    def test_failed_first_run_publishes_nothing(self):
        with patch("pathlib.Path.write_text", side_effect=OSError("metrics write failed")):
            with self.assertRaises(OSError):
                publish_artifacts({"version": "new"}, {"auc": 0.85}, "models")
        self.assertFalse(Path("models/current").exists())
        self.assertEqual(list(Path("models/runs").iterdir()), [])
        self.assertFalse(list(Path("models").glob(".tmp-*")))

    def test_training_write_failure_returns_two_and_preserves_previous(self):
        ref = valid_dataset()
        val = valid_dataset(); val["MonthlyIncome"] += 10000
        self.write_batch(ref, "data/reference/reference.parquet")
        self.write_batch(val, "data/validation/validation.parquet")
        old, snapshot = self.publish_previous()
        with patch("pathlib.Path.write_text", side_effect=OSError("metrics write failed")):
            with self.assertLogs("train", level="ERROR"):
                self.assertEqual(train_main(), 2)
        self.assert_previous_unchanged(old, snapshot)


class MakefileTests(WorkingDirectoryTests):
    def test_demo_requires_contract_exit_and_quarantine_evidence(self):
        driver = Path("driver.py").resolve()
        driver.write_text(
            "import os, sys\nfrom pathlib import Path\n"
            "if sys.argv[2] == 'src.make_bad_batch': sys.exit(0)\n"
            "if os.environ.get('EVIDENCE') == '1':\n"
            " Path('data/quarantine').mkdir(parents=True, exist_ok=True)\n"
            " Path('data/quarantine/bad_batch.parquet').write_bytes(b'batch')\n"
            " Path('data/quarantine/bad_batch_errors.csv').write_text('failure_case\\n15\\n')\n"
            "sys.exit(int(os.environ['INGEST_STATUS']))\n"
        )
        for status, evidence, success in [(0, False, False), (1, False, False),
                                           (1, True, True), (2, True, False)]:
            with self.subTest(status=status, evidence=evidence):
                env = {**os.environ, "INGEST_STATUS": str(status), "EVIDENCE": str(int(evidence))}
                result = subprocess.run(
                    ["make", "-f", str(ROOT / "Makefile"), "demo-bad",
                     f"PYTHON={sys.executable} {driver}"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
