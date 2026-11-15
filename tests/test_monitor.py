"""Integração: treino, simulação, monitoramento, análise causal e dashboard em um
diretório temporário, com dados sintéticos pequenos e o Evidently de verdade."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import config, dashboard, monitor, simulate_batches
from src.causal_analysis import analyse
from src.batch_drift import load_model, score
from src.drift_stats import psi
from src.observability import read_events
from src.train import train
from support import enter_temp_dir, synthetic_credit

MONTHS = ["2026-01", "2026-02", "2026-03"]
UTILIZATION = "RevolvingUtilizationOfUnsecuredLines"


class MonitoringPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enter_temp_dir(cls)
        for path, seed, rows in [(config.REFERENCE, 1, 3000), (Path("data/validation/validation.parquet"), 2, 1000),
                                 (config.HOLDOUT, 3, 3000)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_credit(rows, seed).to_parquet(path)
        cls.baseline_metrics = train()
        # Três meses bastam: controle, choque parcial (com entrega corrompida antes) e choque completo.
        with patch.multiple(simulate_batches, MONTHS=MONTHS, INTENSITY=[0.0, 0.5, 1.0], CORRUPTED_MONTH="2026-02"):
            cls.manifest = simulate_batches.main()
        with patch("src.tracking.start", return_value=None):       # sem MLflow: o pipeline não depende dele
            cls.summary = monitor.run()
        cls.events = read_events()
        cls.batches = {batch["batch"]: batch for batch in cls.summary["batches"]}

    def test_every_batch_is_accounted_for(self):
        self.assertEqual(list(self.batches), ["2026-01", "2026-02-entrega-corrompida", "2026-02", "2026-03"])
        totals = self.summary["totals"]
        self.assertEqual((totals["batches"], totals["success"], totals["rejected"], totals["failed"]), (4, 3, 1, 0))
        self.assertEqual(totals["rows_processed"], 3000)
        self.assertEqual(totals["rows_blocked"], self.batches["2026-02-entrega-corrompida"]["rows"])
        saved = json.loads(config.SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(saved["totals"], totals)

    def test_corrupted_delivery_is_blocked_before_scoring(self):
        bad = self.batches["2026-02-entrega-corrompida"]
        self.assertEqual(bad["status"], "rejected")
        self.assertEqual(list(bad["stages"]), ["ingest"])
        self.assertEqual([(a["rule"], a["severity"]) for a in bad["alerts"]], [("contrato_violado", "critical")])
        self.assertNotIn("drift", bad)
        quarantine = Path(bad["quarantine"])
        self.assertEqual(quarantine.parent, config.QUARANTINE_DIR / self.summary["run_id"])
        self.assertTrue(quarantine.exists())
        self.assertFalse((config.BATCH_DIR / "2026-02-entrega-corrompida.parquet").exists())
        self.assertFalse((config.SCORED_DIR / "2026-02-entrega-corrompida.parquet").exists())
        self.assertEqual({v["column"] for v in bad["violations"]}, {None, "MonthlyIncome", "age", "DebtRatio"})

    def test_control_month_is_quiet_and_shocked_month_alerts(self):
        control, shocked = self.batches["2026-01"], self.batches["2026-03"]
        self.assertEqual([a for a in control["alerts"] if a["rule"].startswith("drift")], [])
        self.assertLess(max(item["psi"] for item in control["drift"].values()), config.PSI_WARNING)
        self.assertEqual(shocked["drift"][UTILIZATION]["status"], "alert")
        self.assertEqual(shocked["drift"]["age"]["status"], "ok")                # variável não alterada
        self.assertIn(("drift_feature", UTILIZATION),
                      [(a["rule"], a.get("column")) for a in shocked["alerts"]])
        self.assertGreater(shocked["performance"]["default_rate"], control["performance"]["default_rate"])
        self.assertGreater(shocked["performance"]["flagged_rate"], control["performance"]["flagged_rate"])

    def test_reported_psi_matches_the_statistic_registered_in_evidently(self):
        model, _ = load_model()
        reference = score(model, pd.read_parquet(config.REFERENCE))
        scored = pd.read_parquet(config.SCORED_DIR / "2026-03.parquet")
        for column in (UTILIZATION, "MonthlyIncome", config.SCORE):
            with self.subTest(column=column):
                self.assertAlmostEqual(self.batches["2026-03"]["drift"][column]["psi"],
                                       psi(reference[column], scored[column]), places=9)

    def test_evidently_report_is_saved_for_each_processed_batch(self):
        for name in MONTHS:
            report = Path(self.batches[name]["report"])
            self.assertEqual(report, config.BATCH_REPORTS / f"{name}.html")
            self.assertGreater(report.stat().st_size, 100_000)
        # O HTML mostra o PSI por decis (mesmo número do resumo), não o PSI nativo do Evidently.
        page = (config.BATCH_REPORTS / "2026-03.html").read_text(encoding="utf-8")
        self.assertTrue("PSI (decis da refer" in page, "método de drift ausente do relatório")
        value = round(self.batches["2026-03"]["drift"][UTILIZATION]["psi"], 6)
        self.assertTrue(f'"drift_score": {value}' in page, f"PSI {value} ausente do relatório")

    def test_baseline_comes_from_the_published_model_version(self):
        model_dir = config.MODEL_DIR.resolve()
        self.assertEqual(self.summary["model_version"], model_dir.name)
        self.assertEqual(self.summary["baseline"]["auc"], self.baseline_metrics["auc"])
        self.assertEqual(self.summary["baseline"]["recall"], self.baseline_metrics[config.RECALL_KEY])
        self.assertEqual(self.summary["baseline"]["rows"], 3000)

    def test_central_log_has_one_event_per_stage_and_alert(self):
        run = [event for event in self.events if event["run_id"] == self.summary["run_id"]]
        stages = [(event["stage"], event["batch"], event["status"]) for event in run]
        for name in MONTHS:
            for stage in ("ingest", "score", "drift", "evaluate"):
                self.assertIn((stage, name, "success"), stages)
        self.assertIn(("ingest", "2026-02-entrega-corrompida", "rejected"), stages)
        self.assertEqual([s for s in stages if s[0] == "monitor"],
                         [("monitor", None, "started"), ("monitor", None, "success")])
        alerts = sum(len(batch["alerts"]) for batch in self.summary["batches"])
        self.assertEqual(sum(event["stage"] == "alert" for event in run), alerts)
        self.assertEqual(self.summary["totals"]["alerts"],
                         {s: sum(a["severity"] == s for b in self.summary["batches"] for a in b["alerts"])
                          for s in ("critical", "alert", "warning")})

    def test_summary_and_log_hold_aggregates_only(self):
        # Nenhuma linha de cliente: o resumo e o log não têm campos por registro.
        income = pd.read_parquet(config.HOLDOUT)["MonthlyIncome"]
        sample = {f"{value:.1f}" for value in income.sample(50, random_state=0)}
        for path in (config.SUMMARY, config.EVENT_LOG):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("failure_case", text)
            self.assertLess(sum(value in text for value in sample), 3, path)   # coincidência numérica, no máximo
        self.assertLess(config.SUMMARY.stat().st_size, 60_000)

    def test_monitor_survives_without_mlflow_and_reports_operational_failure(self):
        entry = {"batch": "2026-09", "month": "2026-09", "intensity": 0.0, "kind": "regular",
                 "file": str(config.BATCH_DIR / "2026-03.parquet")}
        model, _ = load_model()
        reference = score(model, pd.read_parquet(config.REFERENCE))
        with patch("src.monitor.drift_report", side_effect=RuntimeError("evidently fora do ar")):
            with self.assertLogs("monitor", level="ERROR"):
                result = monitor.process_batch(entry, model=model, reference=reference,
                                               baseline=self.summary["baseline"], quarantine_dir="data/quarantine/x")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"]["drift"]["status"], "failed")
        self.assertNotIn("evaluate", result["stages"])
        self.assertEqual([(a["rule"], a["severity"]) for a in result["alerts"]], [("falha_operacional", "critical")])
        self.assertEqual(read_events()[-1]["status"], "failed")

    def test_causal_scenarios_isolate_each_mechanism(self):
        model, _ = load_model()
        reference = score(model, pd.read_parquet(config.REFERENCE))
        raw = simulate_batches.split_months(pd.read_parquet(config.HOLDOUT))[-1]
        result = analyse(model, reference, raw, seed=config.SEED + 2)
        scenarios, parts = result["scenarios"], result["auc_decomposition"]
        self.assertEqual(scenarios["controle"]["effect"]["auc"], 0.0)
        self.assertIsNone(scenarios["controle"]["auc_effect_ci95"])
        self.assertEqual(scenarios["data_drift"]["effect"]["default_rate"], 0.0)       # só X: rótulos intactos
        self.assertEqual(scenarios["concept_drift"]["effect"]["mean_score"], 0.0)      # só Y|X: scores intactos
        self.assertEqual(scenarios["concept_drift"]["effect"]["flagged_rate"], 0.0)
        self.assertEqual(scenarios["concept_drift"]["drifted_features"], [])
        self.assertIn(UTILIZATION, scenarios["data_drift"]["drifted_features"])
        self.assertAlmostEqual(parts["data_drift"] + parts["concept_drift"] + parts["interaction"], parts["total"])
        self.assertEqual(scenarios["ambos"]["metrics"]["mean_score"], scenarios["data_drift"]["metrics"]["mean_score"])
        low, high = scenarios["ambos"]["auc_effect_ci95"]
        self.assertLess(low, high)
        self.assertTrue(low <= parts["total"] <= high)
        self.assertEqual(set(result["feature_interventions"]), set(simulate_batches.SHOCKED_FEATURES))
        self.assertEqual(result["feature_interventions"]["MonthlyIncome"]["effect"]["default_rate"], 0.0)

    def test_dashboard_is_generated_from_the_run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dashboard.main([]), 0)
        page = config.DASHBOARD.read_text(encoding="utf-8")
        self.assertIn(self.summary["run_id"], page)
        self.assertIn("3 de 4", page)
        self.assertIn('href="batches/2026-03.html"', page)               # link relativo para o Evidently
        self.assertIn("mar/26", page)
        self.assertEqual(page.count("<tr data-sev="), sum(len(b["alerts"]) for b in self.summary["batches"]))
        self.assertNotIn("nan", page.lower().replace("financ", ""))


class PerformanceTests(unittest.TestCase):
    def test_metrics_on_a_scored_batch(self):
        scored = pd.DataFrame({config.TARGET: [0, 0, 1, 1], config.SCORE: [0.1, 0.6, 0.4, 0.9]})
        result = monitor.performance(scored)
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertEqual((result["recall"], result["precision"]), (0.5, 0.5))
        self.assertEqual((result["default_rate"], result["flagged_rate"]), (0.5, 0.5))
        self.assertAlmostEqual(result["mean_score"], 0.5)
        self.assertAlmostEqual(result["ks"], 0.5)

    def test_single_class_batch_has_no_auc(self):
        scored = pd.DataFrame({config.TARGET: [0, 0, 0], config.SCORE: [0.1, 0.6, 0.4]})
        result = monitor.performance(scored)
        self.assertIsNone(result["auc"])
        self.assertIsNone(result["ks"])
        self.assertEqual(result["recall"], 0.0)
        self.assertTrue(np.isfinite(result["flagged_rate"]))


if __name__ == "__main__":
    unittest.main()
