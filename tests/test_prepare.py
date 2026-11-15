import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, TARGET
from src.prepare import OUTLIER_COLUMNS, filter_outliers, main, prepare_datasets


class PrepareTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = pd.DataFrame({
            TARGET: np.tile([0, 0, 0, 1], 500),
            "age": rng.integers(18, 90, 2000),
            "MonthlyIncome": np.arange(2000, dtype=float) + 1000,
            "NumberOfDependents": np.zeros(2000),
            "DebtRatio": rng.uniform(0, 10, 2000),
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2, 2000),
        })

    def test_limits_use_only_training_rows(self):
        pool, _ = train_test_split(
            self.raw, test_size=0.3, stratify=self.raw[TARGET], random_state=SEED
        )
        train, _ = train_test_split(
            pool, test_size=0.2, stratify=pool[TARGET], random_state=SEED
        )
        datasets, limits, _ = prepare_datasets(self.raw)
        for column in OUTLIER_COLUMNS:
            self.assertEqual(limits[column], train[column].quantile(0.99))
            for partition in datasets.values():
                self.assertTrue(partition[column].le(limits[column]).all())
        ids = [set(partition.MonthlyIncome) for partition in datasets.values()]
        for i, group in enumerate(ids):
            for other in ids[i + 1:]:
                self.assertFalse(group & other)

    def test_extreme_evaluation_values_do_not_change_training(self):
        original, original_limits, _ = prepare_datasets(self.raw)
        evaluation_ids = pd.concat([
            original["validation"].head(10), original["holdout"].head(10)
        ]).MonthlyIncome
        changed = self.raw.copy()
        changed.loc[changed.MonthlyIncome.isin(evaluation_ids), OUTLIER_COLUMNS] = 1e12
        result, limits, _ = prepare_datasets(changed)
        self.assertEqual(limits, original_limits)
        pd.testing.assert_frame_equal(result["train"], original["train"])
        self.assertLess(len(result["validation"]), len(original["validation"]))
        self.assertLess(len(result["holdout"]), len(original["holdout"]))

    def test_filter_uses_fixed_limits(self):
        df = pd.DataFrame({"DebtRatio": [1.0, 5.0, 1000.0]})
        result, report = filter_outliers(df, {"DebtRatio": 5.0})
        self.assertEqual(result.DebtRatio.tolist(), [1.0, 5.0])
        self.assertEqual(report["outlier DebtRatio > p99"], 1)

    def test_main_creates_outputs_and_persists_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = Path.cwd()
            try:
                os.chdir(temp)
                Path("data/raw").mkdir(parents=True)
                self.raw.to_csv("data/raw/cs-training.csv", index=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
                limits = json.loads(Path("data/reference/outlier_limits.json").read_text())
                report = json.loads(Path("data/reference/cleaning_report.json").read_text())
                for name, path in [
                    ("train", "data/reference/reference.parquet"),
                    ("validation", "data/validation/validation.parquet"),
                    ("holdout", "data/production/holdout.parquet"),
                ]:
                    df = pd.read_parquet(path)
                    self.assertEqual(len(df), report["partitions"][name]["after_outliers"])
                    for column, limit in limits.items():
                        self.assertTrue(df[column].le(limit).all())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
