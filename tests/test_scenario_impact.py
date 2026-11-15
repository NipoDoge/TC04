"""Integração entre os cenários pareados (src.simulate_drift) e as medidas de impacto."""
import json
from pathlib import Path
import unittest

import pandas as pd

from src import config, scenario_impact
from src.observability import read_events
from src.simulate_drift import simulate
from src.train import train
from support import enter_temp_dir, synthetic_credit


class ScenarioImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enter_temp_dir(cls)
        for path, seed, rows in [(config.REFERENCE, 1, 3000), (Path("data/validation/validation.parquet"), 2, 1000),
                                 (config.HOLDOUT, 3, 3000)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_credit(rows, seed).to_parquet(path)
        train()
        simulate(config.HOLDOUT, config.SCENARIO_DIR)             # os quatro cenários da Etapa 2
        cls.before = {path.name: path.read_bytes() for path in config.SCENARIO_DIR.iterdir()}
        cls.summary = scenario_impact.run()
        cls.results = {item["scenario"]: item for item in cls.summary["scenarios"]}

    def test_drift_magnitude_grows_with_the_intervention(self):
        self.assertEqual(list(self.results), list(config.SCENARIOS))
        control, shocked = self.results["t0_control"], self.results["t3"]
        self.assertEqual({item["status"] for item in control["drift"].values()}, {"ok"})
        self.assertEqual(shocked["drift"]["MonthlyIncome"]["status"], "alert")
        self.assertEqual(shocked["drift"]["DebtRatio"]["status"], "alert")
        untouched = [c for c in shocked["drift"] if c not in ("MonthlyIncome", "DebtRatio", config.SCORE)]
        self.assertEqual({shocked["drift"][c]["status"] for c in untouched}, {"ok"})
        for column in ("MonthlyIncome", "DebtRatio"):
            values = [self.results[name]["drift"][column]["psi"] for name in config.SCENARIOS]
            self.assertEqual(values, sorted(values), column)
        self.assertEqual(shocked["factors"], {"MonthlyIncome": 0.60, "DebtRatio": 1.60})
        self.assertEqual(control["factors"], {"MonthlyIncome": 1.0, "DebtRatio": 1.0})

    def test_labels_are_preserved_so_only_the_scores_move(self):
        control, shocked = self.results["t0_control"]["performance"], self.results["t3"]["performance"]
        self.assertEqual(control["default_rate"], shocked["default_rate"])        # rótulos históricos
        self.assertNotEqual(control["mean_score"], shocked["mean_score"])
        self.assertEqual(self.results["t0_control"]["rows"], self.results["t3"]["rows"])

    def test_summary_and_events_are_written_without_touching_the_scenarios(self):
        saved = json.loads(config.SCENARIO_IMPACT.read_text(encoding="utf-8"))
        self.assertEqual(saved["scenarios"], self.summary["scenarios"])
        self.assertEqual(saved["model_version"], config.MODEL_DIR.resolve().name)
        self.assertIn("rótulos históricos preservados", saved["note"])
        after = {path.name: path.read_bytes() for path in config.SCENARIO_DIR.iterdir()}
        self.assertEqual(after, self.before)                                       # entradas intactas
        self.assertFalse(config.SCENARIO_REPORTS.exists())                         # não cria relatórios do Evidently
        events = [(e["stage"], e["batch"], e["status"]) for e in read_events() if e["run_id"] == self.summary["run_id"]]
        self.assertEqual(events, [("scenario", name, "success") for name in config.SCENARIOS])
        income = pd.read_parquet(config.HOLDOUT)["MonthlyIncome"]
        text = config.SCENARIO_IMPACT.read_text(encoding="utf-8")
        self.assertLess(sum(f"{value:.1f}" in text for value in income.sample(50, random_state=0)), 3)

    def test_evidently_verdict_is_read_when_the_reports_exist(self):
        self.assertIsNone(self.results["t3"]["evidently_drifted"])                # src.detect_drift ainda não rodou
        reports = Path("relatorios")
        reports.mkdir()
        (reports / "manifest.json").write_text(json.dumps(
            {"scenarios": [{"name": "t3", "drifted_features": ["MonthlyIncome", "DebtRatio"]}]}), encoding="utf-8")
        summary = scenario_impact.run(report_dir=reports, summary_path="impacto.json", names=("t0_control", "t3"))
        verdicts = {item["scenario"]: item["evidently_drifted"] for item in summary["scenarios"]}
        self.assertEqual(verdicts, {"t0_control": None, "t3": ["MonthlyIncome", "DebtRatio"]})
        self.assertEqual(summary["scenarios"][1]["report"], "relatorios/t3.html")


if __name__ == "__main__":
    unittest.main()
