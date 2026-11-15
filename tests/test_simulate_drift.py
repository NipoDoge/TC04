import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.contract import FEATURE_COLUMNS, schema
from src.artifacts import sha256
from src.config import DATA_DRIFT_FEATURES, LATE90_COLUMN, UTILIZATION_COLUMN
from src.simulate_drift import build_scenarios, simulate


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
        unchanged = original.columns.difference(DATA_DRIFT_FEATURES)
        for name, period in result.items():
            schema.validate(period, lazy=True)
            pd.testing.assert_frame_equal(period[unchanged], original[unchanged])
            if name != "t0_control":
                self.assertTrue(period[UTILIZATION_COLUMN].gt(original[UTILIZATION_COLUMN]).all())
                self.assertTrue((period[LATE90_COLUMN] - original[LATE90_COLUMN]).isin([0, 1]).all())
                self.assertTrue(pd.api.types.is_integer_dtype(period[LATE90_COLUMN]))
        for earlier, later in zip(list(result.values()), list(result.values())[1:]):
            self.assertTrue(later[UTILIZATION_COLUMN].gt(earlier[UTILIZATION_COLUMN]).all())
            self.assertTrue(later[LATE90_COLUMN].ge(earlier[LATE90_COLUMN]).all())

    def test_late_shock_is_reproducible_and_can_change_zero_counts(self):
        self.holdout[LATE90_COLUMN] = 0
        first = build_scenarios(self.holdout, seed=42)
        repeated = build_scenarios(self.holdout, seed=42)
        other = build_scenarios(self.holdout, seed=43)
        for name in first:
            pd.testing.assert_frame_equal(first[name], repeated[name])
        self.assertTrue(first["t3"][LATE90_COLUMN].eq(1).any())
        self.assertFalse(first["t3"][LATE90_COLUMN].equals(other["t3"][LATE90_COLUMN]))

    def test_zero_utilization_is_preserved_by_scaling(self):
        self.holdout.loc[0, UTILIZATION_COLUMN] = 0.0
        for period in build_scenarios(self.holdout).values():
            self.assertEqual(period.loc[0, UTILIZATION_COLUMN], 0.0)

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
