"""Integração: cenários pareados, predições e relatórios de drift lidos pelo resumo de impacto."""
import json
from pathlib import Path
import unittest

import pandas as pd

from src import config, scenario_impact
from src.detect_drift import detect_drift
from src.observability import read_events
from src.predict import predict
from src.simulate_drift import simulate
from src.train import train
from support import enter_temp_dir, synthetic_credit

INCOME, DEBT, UTILIZATION, DELAYS = config.DRIFT_SHIFTED_COLUMNS
PREDICTIONS, REPORTS = Path("data/predictions"), Path("reports/drift")


class ScenarioImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enter_temp_dir(cls)
        for path, seed, rows in [(config.REFERENCE, 1, 3000), (Path("data/validation/validation.parquet"), 2, 1000),
                                 (config.HOLDOUT, 3, 3000)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_credit(rows, seed).to_parquet(path)
        train()
        # A cadeia da Etapa 2: cenários, predições e relatórios de drift, publicados pelos scripts dela.
        simulate(config.HOLDOUT, config.SCENARIO_DIR)
        predict(config.MODEL, config.REFERENCE, config.SCENARIO_DIR, PREDICTIONS)
        detect_drift(config.REFERENCE, config.SCENARIO_DIR, REPORTS)
        inputs = [*config.SCENARIO_DIR.iterdir(), *PREDICTIONS.iterdir(), *REPORTS.iterdir()]
        cls.before = {path: path.read_bytes() for path in inputs}
        cls.summary = scenario_impact.run()
        cls.results = {item["scenario"]: item for item in cls.summary["scenarios"]}

    def test_drift_comes_from_the_published_reports(self):
        self.assertEqual(list(self.results), list(config.SCENARIOS))
        published = pd.read_csv(REPORTS / "summary.csv").set_index(["scenario", "feature"])
        control, shocked = self.results["t0_control"], self.results["t3"]
        self.assertEqual(control["evidently_drifted"], [])
        self.assertIn(INCOME, shocked["evidently_drifted"])
        for feature in config.DRIFT_SHIFTED_COLUMNS:
            self.assertEqual(shocked["drift"][feature]["psi"], published.loc[("t3", feature), "effect_psi"])
        self.assertEqual(shocked["drift"][INCOME]["status"], "alert")
        self.assertEqual({item["status"] for item in control["drift"].values()}, {"ok"})
        self.assertEqual(shocked["factors"], {INCOME: 0.60, DEBT: 1.60, UTILIZATION: 1.60})
        self.assertEqual(shocked["report"], str(REPORTS / "t3.html"))

    def test_score_shift_grows_while_labels_stay_the_same(self):
        rates = [self.results[name]["performance"]["default_rate"] for name in config.SCENARIOS]
        self.assertEqual(len(set(rates)), 1)                                       # rótulos históricos
        shift = [self.results[name]["drift"][config.SCORE]["psi"] for name in config.SCENARIOS]
        self.assertEqual(shift, sorted(shift))
        self.assertGreater(shift[-1], shift[0])
        flagged = [self.results[name]["performance"]["flagged_rate"] for name in config.SCENARIOS]
        self.assertGreater(flagged[-1], flagged[0])                                # o modelo sinaliza mais gente

    def test_summary_and_events_are_written_without_touching_the_inputs(self):
        saved = json.loads(config.SCENARIO_IMPACT.read_text(encoding="utf-8"))
        self.assertEqual(saved["scenarios"], self.summary["scenarios"])
        published = json.loads((PREDICTIONS / "manifest.json").read_text(encoding="utf-8"))["model"]
        self.assertEqual(saved["model"], {"version": Path(published["path"]).parent.name, "sha256": published["sha256"]})
        self.assertNotIn(str(Path.cwd()), config.SCENARIO_IMPACT.read_text(encoding="utf-8"))   # sem caminho da máquina
        self.assertEqual(saved["shifted_columns"], list(config.DRIFT_SHIFTED_COLUMNS))
        self.assertIn("rótulos históricos preservados", saved["note"])
        self.assertEqual({path: path.read_bytes() for path in self.before}, self.before)   # só leitura
        events = [(e["stage"], e["batch"], e["status"]) for e in read_events() if e["run_id"] == self.summary["run_id"]]
        self.assertEqual(events, [("scenario", name, "success") for name in config.SCENARIOS])
        income = pd.read_parquet(config.HOLDOUT)[INCOME]
        text = config.SCENARIO_IMPACT.read_text(encoding="utf-8")
        self.assertLess(sum(f"{value:.1f}" in text for value in income.sample(50, random_state=0)), 3)

    def test_missing_evidence_is_an_error_not_a_silent_recalculation(self):
        with self.assertRaises(FileNotFoundError):
            scenario_impact.run(report_dir="sem-relatorios", summary_path="impacto.json")
        self.assertFalse(Path("impacto.json").exists())


if __name__ == "__main__":
    unittest.main()
