import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.config import DRIFT_SHIFTED_COLUMNS, PERIODS
from src.contract import FEATURE_COLUMNS, schema
from src.artifacts import sha256
from src.simulate_drift import build_scenarios, shift_features, simulate

UTILIZATION, DELAYS = "RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse"


class SimulateDriftTests(unittest.TestCase):
    def setUp(self):
        self.holdout = pd.DataFrame({
            "SeriousDlqin2yrs": np.tile([0, 1], 10),
            **{c: np.arange(20, dtype=float) + 1 for c in FEATURE_COLUMNS},
        })
        for c, column in schema.columns.items():
            if str(column.dtype) == "int64":
                self.holdout[c] = self.holdout[c].astype(int)
        self.holdout["age"] += 18
        self.holdout["MonthlyIncome"] *= 1000

    def test_paired_periods_preserve_source_control_labels_and_other_features(self):
        original = self.holdout.copy(deep=True)
        result = build_scenarios(self.holdout)
        pd.testing.assert_frame_equal(self.holdout, original)
        pd.testing.assert_frame_equal(result["t0_control"], original)
        unchanged = original.columns.difference(list(DRIFT_SHIFTED_COLUMNS))
        for name, period in result.items():
            schema.validate(period, lazy=True)
            pd.testing.assert_frame_equal(period[unchanged], original[unchanged])
            if name != "t0_control":
                self.assertTrue(period.MonthlyIncome.lt(original.MonthlyIncome).all())
                self.assertTrue(period.DebtRatio.gt(original.DebtRatio).all())
                self.assertTrue(period[UTILIZATION].gt(original[UTILIZATION]).all())
        for earlier, later in zip(list(result.values()), list(result.values())[1:]):
            self.assertTrue(later.MonthlyIncome.lt(earlier.MonthlyIncome).all())
            self.assertTrue(later.DebtRatio.gt(earlier.DebtRatio).all())
            self.assertTrue(later[UTILIZATION].gt(earlier[UTILIZATION]).all())
            self.assertTrue(later[DELAYS].ge(earlier[DELAYS]).all())

    def test_delay_is_added_only_to_clients_who_reach_the_credit_limit(self):
        # Utilização de 0,50 a 1,45: quem está abaixo de 1 e cruza o limite com o fator ganha um atraso.
        self.holdout[UTILIZATION] = np.linspace(0.50, 1.45, len(self.holdout))
        original = self.holdout.copy(deep=True)
        result = build_scenarios(self.holdout)
        pd.testing.assert_frame_equal(result["t0_control"], original)
        previous = 0
        for name, _, _, factor in PERIODS[1:]:
            crossed = original[UTILIZATION].lt(1) & (original[UTILIZATION] * factor).ge(1)
            added = result[name][DELAYS] - original[DELAYS]
            self.assertTrue(added.eq(crossed.astype(int)).all(), name)
            self.assertFalse(added[original[UTILIZATION].ge(1)].any(), name)      # já no limite: sem atraso novo
            self.assertGreater(int(crossed.sum()), previous, name)               # mais gente a cada período
            self.assertEqual(str(result[name][DELAYS].dtype), "int64")
            previous = int(crossed.sum())

    def test_one_variable_can_be_shifted_at_a_time(self):
        self.holdout[UTILIZATION] = np.linspace(0.50, 1.45, len(self.holdout))
        _, income, debt, utilization = PERIODS[-1]
        full = shift_features(self.holdout, income, debt, utilization)
        for column in DRIFT_SHIFTED_COLUMNS:
            only = shift_features(self.holdout, income, debt, utilization, columns=(column,))
            others = self.holdout.columns.difference([column])
            pd.testing.assert_frame_equal(only[others], self.holdout[others])
            pd.testing.assert_series_equal(only[column], full[column])
        untouched = shift_features(self.holdout, income, debt, utilization, columns=())
        pd.testing.assert_frame_equal(untouched, self.holdout)

    def test_publication_records_provenance_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference = root / "holdout.parquet", root / "reference.parquet"
            self.holdout.to_parquet(source)
            self.holdout.iloc[:10].to_parquet(reference)
            before = {p: sha256(p) for p in [source, reference]}
            output = root / "scenarios"
            manifest = simulate(source, output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
            self.assertIn("não fornece uma série temporal real", manifest["limitation"])
            self.assertEqual(manifest["source"]["sha256"], before[source])
            for period in manifest["periods"]:
                path = output / period["file"]
                self.assertEqual(period["sha256"], sha256(path))
                self.assertEqual(len(pd.read_parquet(path)), len(self.holdout))
                self.assertEqual(set(period["factors"]), set(DRIFT_SHIFTED_COLUMNS) - {DELAYS})
                self.assertEqual(period["rows_with_new_delay"], 0)      # no exemplo todos já estão no limite
            self.assertIn("recebe +1", manifest["delay_rule"])
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaises(FileExistsError):
                simulate(source, output)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))

    def test_invalid_holdout_does_not_publish_scenarios(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "holdout.parquet", Path(temp) / "scenarios"
            self.holdout.loc[0, "MonthlyIncome"] = -1
            self.holdout.to_parquet(source)
            with self.assertRaises(pa.errors.SchemaErrors):
                simulate(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
