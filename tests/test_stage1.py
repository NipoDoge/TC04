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

from src.ingest import ingest
from src.train import TrainingDataError, main as train_main, train

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
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        for path, expected in [(str(good), 0), ("bad.parquet", 1), ("missing.parquet", 2)]:
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
                        with self.assertRaises(TrainingDataError):
                            train()
                        model.assert_not_called()
                    self.assertFalse(Path("models").exists())

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
