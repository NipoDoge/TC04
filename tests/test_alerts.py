import unittest

from src import config
from src.alerts import count_by_severity, describe_violation, evaluate

BASELINE = {"auc": 0.86, "recall": 0.77, "default_rate": 0.066}
HEALTHY = {"auc": 0.85, "recall": 0.76, "default_rate": 0.07}


def drift(**psi_by_column):
    status = lambda v: "alert" if v >= config.PSI_ALERT else "warning" if v >= config.PSI_WARNING else "ok"
    return {column: {"psi": value, "status": status(value)} for column, value in psi_by_column.items()}


def rules(alerts):
    return [(alert["rule"], alert["severity"]) for alert in alerts]


class AlertRuleTests(unittest.TestCase):
    def test_healthy_batch_raises_nothing(self):
        alerts = evaluate("2026-01", rows=4000, drift=drift(age=0.01, score=0.02), performance=HEALTHY,
                          baseline=BASELINE, duration_s=1.2, previous_durations=[1.1, 1.3, 1.2])
        self.assertEqual(alerts, [])

    def test_contract_violation_is_critical_and_stops_other_rules(self):
        violations = [{"column": None, "check": "linhas duplicadas", "rows": 400},
                      {"column": "MonthlyIncome", "check": "not_nullable", "rows": 120},
                      {"column": "age", "check": "in_range(18, 100)", "rows": 40}]
        alerts = evaluate("lote", ingest_code=1, violations=violations, rows=10, drift=drift(age=0.9))
        self.assertEqual(rules(alerts), [("contrato_violado", "critical")])
        message = alerts[0]["message"]
        for expected in ["linhas duplicadas (400 linhas)", "MonthlyIncome: valor nulo (120 linhas)",
                         "age: fora do intervalo de 18 a 100 (40 linhas)"]:
            self.assertIn(expected, message)

    def test_operational_failure_is_critical(self):
        alerts = evaluate("lote", ingest_code=2, error="leitura: arquivo ausente")
        self.assertEqual(rules(alerts), [("falha_operacional", "critical")])
        self.assertIn("arquivo ausente", alerts[0]["message"])

    def test_low_volume_warns_below_the_minimum_only(self):
        self.assertEqual(rules(evaluate("lote", rows=config.MIN_ROWS - 1)), [("volume_baixo", "warning")])
        self.assertEqual(evaluate("lote", rows=config.MIN_ROWS), [])

    def test_feature_drift_follows_psi_thresholds(self):
        alerts = evaluate("lote", drift=drift(a=config.PSI_WARNING - 0.001, b=config.PSI_WARNING,
                                             c=config.PSI_ALERT))
        self.assertEqual(rules(alerts), [("drift_feature", "warning"), ("drift_feature", "alert")])
        self.assertEqual([alert["column"] for alert in alerts], ["b", "c"])
        self.assertEqual([alert["threshold"] for alert in alerts], [config.PSI_WARNING, config.PSI_ALERT])

    def test_score_drift_has_its_own_rule(self):
        alerts = evaluate("lote", drift=drift(**{config.SCORE: 0.3}))
        self.assertEqual(rules(alerts), [("drift_predicao", "alert")])
        self.assertIn("score do modelo", alerts[0]["message"])

    def test_auc_drop_alerts_at_the_threshold(self):
        ok = {**HEALTHY, "auc": BASELINE["auc"] - config.AUC_DROP + 0.001}
        bad = {**HEALTHY, "auc": BASELINE["auc"] - config.AUC_DROP - 0.001}
        self.assertEqual(evaluate("lote", performance=ok, baseline=BASELINE), [])
        self.assertEqual(rules(evaluate("lote", performance=bad, baseline=BASELINE)), [("queda_auc", "alert")])

    def test_missing_auc_does_not_break_the_other_rules(self):
        # Lote com uma só classe: AUC indefinida, as demais regras continuam valendo.
        performance = {"auc": None, "recall": 0.5, "default_rate": 0.07}
        self.assertEqual(rules(evaluate("lote", performance=performance, baseline=BASELINE)),
                         [("queda_recall", "alert")])

    def test_recall_drop_alerts_at_the_threshold(self):
        ok = {**HEALTHY, "recall": BASELINE["recall"] - config.RECALL_DROP + 0.001}
        bad = {**HEALTHY, "recall": BASELINE["recall"] - config.RECALL_DROP - 0.001}
        self.assertEqual(evaluate("lote", performance=ok, baseline=BASELINE), [])
        self.assertEqual(rules(evaluate("lote", performance=bad, baseline=BASELINE)), [("queda_recall", "alert")])

    def test_default_rate_alerts_relative_to_reference(self):
        limit = BASELINE["default_rate"] * config.DEFAULT_RATE_FACTOR
        self.assertEqual(evaluate("lote", performance={**HEALTHY, "default_rate": limit - 0.001}, baseline=BASELINE), [])
        alerts = evaluate("lote", performance={**HEALTHY, "default_rate": limit + 0.001}, baseline=BASELINE)
        self.assertEqual(rules(alerts), [("inadimplencia_elevada", "alert")])
        self.assertAlmostEqual(alerts[0]["threshold"], limit)

    def test_latency_needs_history_and_compares_with_the_median(self):
        history = [1.0, 1.2, 1.1]
        self.assertEqual(evaluate("lote", duration_s=9.0, previous_durations=history[:2]), [])
        self.assertEqual(evaluate("lote", duration_s=config.LATENCY_FACTOR * 1.1, previous_durations=history), [])
        alerts = evaluate("lote", duration_s=config.LATENCY_FACTOR * 1.1 + 0.1, previous_durations=history)
        self.assertEqual(rules(alerts), [("latencia_elevada", "warning")])

    def test_every_alert_carries_batch_and_no_customer_rows(self):
        alerts = evaluate("2026-08", rows=100, drift=drift(a=0.5, score=0.5),
                          performance={"auc": 0.6, "recall": 0.5, "default_rate": 0.2}, baseline=BASELINE,
                          duration_s=10, previous_durations=[1, 1, 1])
        self.assertEqual(len(alerts), 7)
        for alert in alerts:
            self.assertEqual(alert["batch"], "2026-08")
            self.assertLessEqual(set(alert), {"batch", "severity", "rule", "message", "column", "value", "threshold"})
        self.assertEqual(count_by_severity(alerts), {"critical": 0, "alert": 5, "warning": 2})


class ViolationLabelTests(unittest.TestCase):
    def test_contract_checks_are_described_in_plain_language(self):
        cases = {"not_nullable": "valor nulo", "greater_than_or_equal_to(0)": "menor que 0",
                 "isin([0, 1])": "fora dos valores [0, 1]", "finite": "valor infinito",
                 "column_in_dataframe": "coluna ausente", "regra_desconhecida": "regra_desconhecida"}
        for check, expected in cases.items():
            with self.subTest(check=check):
                self.assertEqual(describe_violation({"column": "x", "check": check, "rows": 3}),
                                 f"x: {expected} (3 linhas)")
        self.assertEqual(describe_violation({"column": None, "check": "non_empty", "rows": 1}), "lote vazio (1 linhas)")


if __name__ == "__main__":
    unittest.main()
