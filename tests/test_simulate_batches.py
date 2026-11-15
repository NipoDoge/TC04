import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src import config
from src.contract import schema
from src.observability import read_events
from src.simulate_batches import (CORRUPTED_MONTH, DEBT, DELAYS, INCOME, INTENSITY, MONTHS, SHOCKED_FEATURES,
                                UTILIZATION, apply_shock, build_batches, corrupt, main, split_months)
from support import TempDirTestCase, synthetic_credit


class ShockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_credit(4000, seed=1)

    def test_zero_intensity_changes_nothing_and_returns_a_copy(self):
        result = apply_shock(self.data, 0.0, seed=1)
        pd.testing.assert_frame_equal(result, self.data)
        self.assertIsNot(result, self.data)

    def test_same_seed_is_reproducible_and_seed_matters(self):
        first, again = apply_shock(self.data, 1.0, seed=7), apply_shock(self.data, 1.0, seed=7)
        pd.testing.assert_frame_equal(first, again)
        self.assertFalse(first.equals(apply_shock(self.data, 1.0, seed=8)))

    def test_only_documented_columns_change_and_input_is_preserved(self):
        original = self.data.copy()
        shocked = apply_shock(self.data, 1.0, seed=3)
        pd.testing.assert_frame_equal(self.data, original)
        untouched = [c for c in self.data.columns if c not in (*SHOCKED_FEATURES, config.TARGET)]
        pd.testing.assert_frame_equal(shocked[untouched], self.data[untouched])
        for column in (*SHOCKED_FEATURES, config.TARGET):
            self.assertFalse(shocked[column].equals(self.data[column]), column)

    def test_shock_moves_each_variable_in_the_documented_direction(self):
        shocked = apply_shock(self.data, 1.0, seed=3)
        self.assertTrue((shocked[INCOME] <= self.data[INCOME]).all())           # renda cai
        self.assertTrue((shocked[DEBT] >= self.data[DEBT]).all())               # comprometimento sobe
        self.assertTrue((shocked[UTILIZATION] >= self.data[UTILIZATION]).all())  # uso do rotativo sobe
        self.assertTrue((shocked[DELAYS] >= self.data[DELAYS]).all())           # novos atrasos

    def test_labels_only_change_from_good_to_bad(self):
        shocked = apply_shock(self.data, 1.0, seed=3)
        changed = shocked[config.TARGET] != self.data[config.TARGET]
        self.assertGreater(changed.sum(), 0)
        self.assertTrue((self.data.loc[changed, config.TARGET] == 0).all())
        self.assertTrue((shocked.loc[changed, config.TARGET] == 1).all())

    def test_effect_grows_with_intensity(self):
        incomes, defaults = [], []
        for t in (0.0, 0.4, 1.0):
            shocked = apply_shock(self.data, t, seed=3)
            incomes.append(shocked[INCOME].mean())
            defaults.append(shocked[config.TARGET].mean())
        self.assertEqual(incomes, sorted(incomes, reverse=True))
        self.assertEqual(defaults, sorted(defaults))

    def test_mechanisms_can_be_applied_in_isolation(self):
        features = [c for c in self.data.columns if c != config.TARGET]
        only_x = apply_shock(self.data, 1.0, seed=3, concept=False)
        pd.testing.assert_series_equal(only_x[config.TARGET], self.data[config.TARGET])
        only_y = apply_shock(self.data, 1.0, seed=3, features=())
        pd.testing.assert_frame_equal(only_y[features], self.data[features])
        both = apply_shock(self.data, 1.0, seed=3)
        pd.testing.assert_frame_equal(both[features], only_x[features])          # mesmo sorteio
        pd.testing.assert_series_equal(both[config.TARGET], only_y[config.TARGET])
        one = apply_shock(self.data, 1.0, seed=3, features=(INCOME,), concept=False)
        others = [c for c in self.data.columns if c != INCOME]
        pd.testing.assert_frame_equal(one[others], self.data[others])


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holdout = synthetic_credit(4000, seed=2)
        cls.batches = build_batches(cls.holdout, seed=11)

    def test_months_partition_the_holdout_without_overlap(self):
        parts = split_months(self.holdout, seed=11)
        self.assertEqual(len(parts), len(MONTHS))
        self.assertEqual(sum(len(part) for part in parts), len(self.holdout))
        merged = pd.concat(parts).sort_values(list(self.holdout.columns)).reset_index(drop=True)
        expected = self.holdout.sort_values(list(self.holdout.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(merged, expected)

    def test_batches_arrive_in_order_with_the_corrupted_delivery_first(self):
        names = [meta["batch"] for meta, _ in self.batches]
        position = MONTHS.index(CORRUPTED_MONTH)
        expected = MONTHS[:position] + [f"{CORRUPTED_MONTH}-entrega-corrompida"] + MONTHS[position:]
        self.assertEqual(names, expected)
        regular = [meta for meta, _ in self.batches if meta["kind"] == "regular"]
        self.assertEqual([meta["intensity"] for meta in regular], INTENSITY)
        for meta, df in self.batches:
            self.assertEqual(meta["rows"], len(df))

    def test_control_months_are_untouched_slices_of_the_holdout(self):
        parts = split_months(self.holdout, seed=11)
        for (meta, df), part in zip([b for b in self.batches if b[0]["kind"] == "regular"], parts):
            if meta["intensity"] == 0:
                pd.testing.assert_frame_equal(df, part)

    def test_drifted_batches_still_satisfy_the_data_contract(self):
        # Drift é dado válido que mudou de distribuição: o contrato não pode barrá-lo.
        for meta, df in self.batches:
            if meta["kind"] == "regular":
                with self.subTest(batch=meta["batch"]):
                    schema.validate(df, lazy=True)

    def test_corrupted_delivery_violates_the_contract(self):
        meta, bad = next(batch for batch in self.batches if batch[0]["kind"] == "corrupted")
        with self.assertRaises(pa.errors.SchemaErrors) as caught:
            schema.validate(bad, lazy=True)
        failures = caught.exception.failure_cases
        self.assertTrue(((failures["column"] == INCOME) & (failures["check"] == "not_nullable")).any())
        self.assertTrue(((failures["column"] == "age") & failures["check"].str.contains("in_range")).any())
        self.assertTrue(((failures["column"] == DEBT) & failures["check"].str.contains("greater_than")).any())
        self.assertTrue(failures["check"].str.contains("uniqueness").any())

    def test_corruption_is_reproducible_and_keeps_the_original_rows(self):
        clean = self.holdout.head(600).reset_index(drop=True)
        bad = corrupt(clean, seed=5)
        pd.testing.assert_frame_equal(bad, corrupt(clean, seed=5))
        self.assertEqual(len(bad), len(clean) + 200)                 # reenvio duplicado
        self.assertEqual(bad[INCOME].isna().sum(), 120 + bad.iloc[len(clean):][INCOME].isna().sum())
        self.assertEqual((bad["age"].head(len(clean)) == 0).sum(), 40)
        self.assertEqual((bad[DEBT].head(len(clean)) < 0).sum(), 20)


class SimulationOutputTests(TempDirTestCase):
    def test_main_writes_batches_manifest_and_event(self):
        holdout = Path("holdout.parquet")
        synthetic_credit(4000, seed=2).to_parquet(holdout)
        Path("lotes").mkdir()
        Path("lotes/antigo.parquet").write_bytes(b"lote de uma execucao anterior")
        manifest = main(holdout, "lotes", seed=11)
        saved = json.loads(Path("lotes/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)
        self.assertEqual(saved["seed"], 11)
        self.assertEqual(saved["shocked_features"], list(SHOCKED_FEATURES))
        self.assertEqual(len(saved["batches"]), len(MONTHS) + 1)
        self.assertFalse(Path("lotes/antigo.parquet").exists())      # lotes antigos não se misturam
        for entry in saved["batches"]:
            self.assertEqual(len(pd.read_parquet(entry["file"])), entry["rows"])
        event = read_events()[-1]
        self.assertEqual((event["stage"], event["status"]), ("simulate", "success"))
        self.assertEqual(event["rows"], sum(entry["rows"] for entry in saved["batches"]))

    def test_two_runs_with_the_same_seed_write_identical_batches(self):
        holdout = Path("holdout.parquet")
        synthetic_credit(4000, seed=2).to_parquet(holdout)
        first = {e["batch"]: pd.read_parquet(e["file"]) for e in main(holdout, "a", seed=11)["batches"]}
        second = {e["batch"]: pd.read_parquet(e["file"]) for e in main(holdout, "b", seed=11)["batches"]}
        self.assertEqual(list(first), list(second))
        for name, df in first.items():
            pd.testing.assert_frame_equal(df, second[name])


if __name__ == "__main__":
    unittest.main()
