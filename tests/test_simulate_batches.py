import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src import config
from src.concept_drift import build_concept_scenarios
from src.contract import schema
from src.observability import read_events
from src.simulate_batches import apply_levels, build_batches, corrupt, income_cutoff, main, split_months
from src.simulate_drift import build_scenarios
from support import TempDirTestCase, synthetic_credit

INCOME, DEBT, UTILIZATION, DELAYS = config.DRIFT_SHIFTED_COLUMNS
MONTHS = [month for month, _, _ in config.MONTHLY_SCHEDULE]


class LevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = synthetic_credit(4000, seed=0)
        cls.data = synthetic_credit(4000, seed=1)
        cls.cutoff = income_cutoff(cls.reference)

    def shock(self, data_level="t3", concept_level="c3", seed=3, **options):
        return apply_levels(self.data, data_level, concept_level, self.cutoff, seed, **options)

    def test_control_levels_change_nothing_and_return_a_copy(self):
        result = self.shock("t0_control", "c0_control")
        pd.testing.assert_frame_equal(result, self.data)
        self.assertIsNot(result, self.data)

    def test_same_seed_is_reproducible_and_seed_only_affects_labels(self):
        first, again, other = self.shock(seed=7), self.shock(seed=7), self.shock(seed=8)
        pd.testing.assert_frame_equal(first, again)
        features = [c for c in self.data.columns if c != config.TARGET]
        pd.testing.assert_frame_equal(first[features], other[features])           # data drift não tem sorteio
        self.assertFalse(first[config.TARGET].equals(other[config.TARGET]))

    def test_levels_are_the_mechanisms_of_the_paired_experiments(self):
        # Data drift: igual ao cenário pareado do mesmo nível. Concept drift: mesma regra, com o sorteio do lote.
        paired = build_scenarios(self.data)
        features = [c for c in self.data.columns if c != config.TARGET]
        for level in config.SCENARIOS:
            pd.testing.assert_frame_equal(self.shock(level, "c0_control")[features], paired[level][features])
        concept, _ = build_concept_scenarios(self.data, self.cutoff, seed=3)
        for level, _ in config.CONCEPT_PERIODS:
            pd.testing.assert_series_equal(self.shock("t0_control", level)[config.TARGET], concept[level][config.TARGET])

    def test_only_documented_columns_change_and_input_is_preserved(self):
        original = self.data.copy()
        shocked = self.shock()
        pd.testing.assert_frame_equal(self.data, original)
        untouched = [c for c in self.data.columns if c not in (*config.DRIFT_SHIFTED_COLUMNS, config.TARGET)]
        pd.testing.assert_frame_equal(shocked[untouched], self.data[untouched])
        for column in (*config.DRIFT_SHIFTED_COLUMNS, config.TARGET):
            self.assertFalse(shocked[column].equals(self.data[column]), column)
        self.assertTrue((shocked[INCOME] <= self.data[INCOME]).all())            # renda cai
        self.assertTrue((shocked[DEBT] >= self.data[DEBT]).all())                # endividamento sobe
        self.assertTrue((shocked[UTILIZATION] >= self.data[UTILIZATION]).all())  # uso do rotativo sobe
        self.assertTrue((shocked[DELAYS] >= self.data[DELAYS]).all())            # novos atrasos

    def test_labels_only_change_from_good_to_bad_inside_the_low_income_segment(self):
        shocked = self.shock()
        changed = shocked[config.TARGET] != self.data[config.TARGET]
        self.assertGreater(changed.sum(), 0)
        self.assertTrue((self.data.loc[changed, config.TARGET] == 0).all())
        self.assertTrue((shocked.loc[changed, config.TARGET] == 1).all())
        self.assertTrue((self.data.loc[changed, INCOME] <= self.cutoff).all())    # renda original, antes do choque

    def test_higher_levels_shift_more(self):
        incomes = [self.shock(level, "c0_control")[INCOME].mean() for level in config.SCENARIOS]
        defaults = [self.shock("t0_control", level)[config.TARGET].mean() for level, _ in config.CONCEPT_PERIODS]
        self.assertEqual(incomes, sorted(incomes, reverse=True))
        self.assertEqual(defaults, sorted(defaults))

    def test_mechanisms_can_be_applied_in_isolation(self):
        features = [c for c in self.data.columns if c != config.TARGET]
        only_x, only_y, both = self.shock(concept=False), self.shock(columns=()), self.shock()
        pd.testing.assert_series_equal(only_x[config.TARGET], self.data[config.TARGET])
        pd.testing.assert_frame_equal(only_y[features], self.data[features])
        pd.testing.assert_frame_equal(both[features], only_x[features])
        pd.testing.assert_series_equal(both[config.TARGET], only_y[config.TARGET])   # mesmo sorteio
        one = self.shock(columns=(INCOME,), concept=False)
        others = [c for c in self.data.columns if c != INCOME]
        pd.testing.assert_frame_equal(one[others], self.data[others])


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holdout = synthetic_credit(4000, seed=2)
        cls.cutoff = income_cutoff(synthetic_credit(4000, seed=0))
        cls.batches = build_batches(cls.holdout, cls.cutoff, seed=11)

    def test_months_partition_the_holdout_without_overlap(self):
        parts = split_months(self.holdout, seed=11)
        self.assertEqual(len(parts), len(MONTHS))
        self.assertEqual(sum(len(part) for part in parts), len(self.holdout))
        merged = pd.concat(parts).sort_values(list(self.holdout.columns)).reset_index(drop=True)
        expected = self.holdout.sort_values(list(self.holdout.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(merged, expected)

    def test_batches_follow_the_schedule_with_the_corrupted_delivery_first(self):
        names = [meta["batch"] for meta, _ in self.batches]
        position = MONTHS.index(config.CORRUPTED_MONTH)
        expected = MONTHS[:position] + [f"{config.CORRUPTED_MONTH}-entrega-corrompida"] + MONTHS[position:]
        self.assertEqual(names, expected)
        regular = [meta for meta, _ in self.batches if meta["kind"] == "regular"]
        self.assertEqual([(m["month"], m["data_level"], m["concept_level"]) for m in regular],
                         list(config.MONTHLY_SCHEDULE))
        for meta, df in self.batches:
            self.assertEqual(meta["rows"], len(df))

    def test_control_months_are_untouched_slices_of_the_holdout(self):
        parts = split_months(self.holdout, seed=11)
        checked = 0
        for (meta, df), part in zip([b for b in self.batches if b[0]["kind"] == "regular"], parts):
            if (meta["data_level"], meta["concept_level"]) == ("t0_control", "c0_control"):
                pd.testing.assert_frame_equal(df, part)
                checked += 1
        self.assertGreaterEqual(checked, 1)

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
        self.assertEqual(bad[INCOME].head(len(clean)).isna().sum(), 120)
        self.assertEqual((bad["age"].head(len(clean)) == 0).sum(), 40)
        self.assertEqual((bad[DEBT].head(len(clean)) < 0).sum(), 20)


class SimulationOutputTests(TempDirTestCase):
    def write_inputs(self):
        holdout, reference = Path("holdout.parquet"), Path("reference.parquet")
        synthetic_credit(4000, seed=2).to_parquet(holdout)
        synthetic_credit(4000, seed=0).to_parquet(reference)
        return holdout, reference

    def test_main_writes_batches_manifest_and_event(self):
        holdout, reference = self.write_inputs()
        Path("lotes").mkdir()
        Path("lotes/antigo.parquet").write_bytes(b"lote de uma execucao anterior")
        manifest = main(holdout, "lotes", seed=11, reference_path=reference)
        saved = json.loads(Path("lotes/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)
        self.assertEqual(saved["seed"], 11)
        self.assertEqual(saved["shocked_features"], list(config.DRIFT_SHIFTED_COLUMNS))
        self.assertEqual([tuple(item.values()) for item in saved["schedule"]], list(config.MONTHLY_SCHEDULE))
        self.assertEqual(saved["concept_drift_levels"], dict(config.CONCEPT_PERIODS))
        self.assertEqual(set(saved["data_drift_levels"]), set(config.SCENARIOS))
        self.assertEqual(saved["income_cutoff"], income_cutoff(pd.read_parquet(reference)))
        self.assertEqual(len(saved["batches"]), len(MONTHS) + 1)
        self.assertFalse(Path("lotes/antigo.parquet").exists())      # lotes antigos não se misturam
        for entry in saved["batches"]:
            self.assertEqual(len(pd.read_parquet(entry["file"])), entry["rows"])
        event = read_events()[-1]
        self.assertEqual((event["stage"], event["status"]), ("simulate", "success"))
        self.assertEqual(event["rows"], sum(entry["rows"] for entry in saved["batches"]))

    def test_two_runs_with_the_same_seed_write_identical_batches(self):
        holdout, reference = self.write_inputs()
        first = {e["batch"]: pd.read_parquet(e["file"]) for e in main(holdout, "a", 11, reference)["batches"]}
        second = {e["batch"]: pd.read_parquet(e["file"]) for e in main(holdout, "b", 11, reference)["batches"]}
        self.assertEqual(list(first), list(second))
        for name, df in first.items():
            pd.testing.assert_frame_equal(df, second[name])

    def test_schedule_and_corrupted_month_come_from_config(self):
        holdout, reference = self.write_inputs()
        schedule = (("2027-01", "t0_control", "c0_control"), ("2027-02", "t3", "c2"))
        with patch.object(config, "MONTHLY_SCHEDULE", schedule), patch.object(config, "CORRUPTED_MONTH", "2027-02"):
            manifest = main(holdout, "curto", 11, reference)
        self.assertEqual([e["batch"] for e in manifest["batches"]], ["2027-01", "2027-02-entrega-corrompida", "2027-02"])
        self.assertEqual([e["rows"] for e in manifest["batches"] if e["kind"] == "regular"], [2000, 2000])


if __name__ == "__main__":
    unittest.main()
