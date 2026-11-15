import contextlib
import html
import io
import json
import re
import unittest

from src import config
from src.dashboard import (batch_label, build, line_chart, main, nice_ticks, num, pct, psi_level, signed)
from src.batch_drift import status_for
from support import TempDirTestCase

THRESHOLDS = {"psi_warning": config.PSI_WARNING, "psi_alert": config.PSI_ALERT, "auc_drop": config.AUC_DROP,
              "recall_drop": config.RECALL_DROP, "default_rate_factor": config.DEFAULT_RATE_FACTOR,
              "min_rows": config.MIN_ROWS, "latency_factor": config.LATENCY_FACTOR,
              "decision_threshold": config.THRESHOLD}
BASELINE = {"auc": 0.857, "recall": 0.772, "ks": 0.568, "default_rate": 0.066, "mean_score": 0.327,
            "flagged_rate": 0.246, "rows": 65867}


def good_batch(name, auc, psi_income, alerts=()):
    drift = {column: {"psi": value, "ks_stat": 0.1, "ks_pvalue": 0.01, "status": status_for(value),
                      "ref_mean": 1.0, "cur_mean": 1.2, "ref_median": 1.0, "cur_median": 1.1}
             for column, value in {"MonthlyIncome": psi_income, "age": 0.002, config.SCORE: psi_income / 2}.items()}
    stages = {stage: {"status": "success", "duration_s": 0.25} for stage in ("ingest", "score", "drift", "evaluate")}
    return {"batch": name, "month": name, "intensity": 0.0, "kind": "regular", "status": "success", "rows": 4412,
            "duration_s": 1.0, "stages": stages, "alerts": list(alerts), "drift": drift,
            "performance": {"auc": auc, "ks": 0.5, "recall": 0.75, "precision": 0.2, "default_rate": 0.07,
                            "mean_score": 0.33, "flagged_rate": 0.25}}


def rejected_batch(name="2026-02-entrega-corrompida"):
    alert = {"batch": name, "severity": "critical", "rule": "contrato_violado",
             "message": "Lote bloqueado pelo contrato de dados: age: valor nulo (3 linhas)"}
    return {"batch": name, "month": "2026-02", "intensity": 0.0, "kind": "corrupted", "status": "rejected",
            "rows": 4612, "duration_s": 0.1, "stages": {"ingest": {"status": "rejected", "duration_s": 0.1}},
            "alerts": [alert], "violations": [{"column": "age", "check": "not_nullable", "rows": 3}]}


def make_summary(batches):
    alerts = [alert for batch in batches for alert in batch["alerts"]]
    done = [batch for batch in batches if batch["status"] == "success"]
    return {
        "run_id": "mon-teste", "generated_at": "2026-11-14T23:41:49+00:00", "model_version": "abc123",
        "features": ["MonthlyIncome", "age"], "baseline": BASELINE, "thresholds": THRESHOLDS,
        "simulation": {"seed": 42, "source": "data/production/holdout.parquet", "parameters": {"stress_share": 0.5},
                       "shocked_features": ["MonthlyIncome"]},
        "totals": {"batches": len(batches), "success": len(done), "rejected": len(batches) - len(done), "failed": 0,
                   "rows_processed": sum(b["rows"] for b in done),
                   "rows_blocked": sum(b["rows"] for b in batches if b["status"] != "success"),
                   "median_duration_s": 1.0 if done else None,
                   "alerts": {s: sum(a["severity"] == s for a in alerts) for s in ("critical", "alert", "warning")}},
        "batches": batches,
    }


def drift_alert(batch, severity="alert"):
    return {"batch": batch, "severity": severity, "rule": "drift_feature", "column": "MonthlyIncome",
            "message": "Drift em MonthlyIncome: PSI 0,300 (limite 0,20)", "value": 0.3, "threshold": 0.2}


def auc_alert(batch):
    return {"batch": batch, "severity": "alert", "rule": "queda_auc", "value": 0.7, "threshold": 0.807,
            "message": "AUC caiu de 0,857 para 0,700 (queda de 0,157, limite 0,05)"}


FULL = [good_batch("2026-01", 0.85, 0.01), rejected_batch(),
        good_batch("2026-02", 0.70, 0.05, [auc_alert("2026-02")]),
        good_batch("2026-03", 0.65, 0.30, [auc_alert("2026-03"), drift_alert("2026-03")])]


class FormattingTests(unittest.TestCase):
    def test_numbers_follow_brazilian_format(self):
        self.assertEqual(num(35297, 0), "35.297")
        self.assertEqual(num(0.8572), "0,857")
        self.assertEqual(pct(0.1387), "13,9%")
        self.assertEqual(signed(-0.0635), "−0,064")
        self.assertEqual(signed(0.1143), "+0,114")
        self.assertEqual((num(None), pct(None), signed(None)), ("—", "—", "—"))

    def test_batch_labels(self):
        self.assertEqual(batch_label("2026-05"), "mai/26")
        self.assertEqual(batch_label("2026-05-entrega-corrompida"), "mai/26, entrega corrompida")
        self.assertEqual(batch_label("lote-avulso"), "lote-avulso")

    def test_ticks_cover_the_range_with_even_steps(self):
        for low, high in [(0.691, 0.857), (0.0, 0.429), (0.0, 0.502), (0.66, 0.78), (3, 3)]:
            with self.subTest(low=low, high=high):
                ticks = nice_ticks(low, high)
                self.assertLessEqual(ticks[0], low + 1e-9)
                self.assertGreaterEqual(ticks[-1], high - 1e-9)
                steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
                self.assertEqual(len(steps), 1)                  # inclui a série constante (3, 3)
                self.assertGreater(ticks[-1], ticks[0])
                self.assertLessEqual(len(ticks), 12)

    def test_psi_colour_levels_follow_the_alert_thresholds(self):
        cases = [(0.0, 1), (0.049, 1), (0.05, 2), (0.099, 2), (0.10, 3), (0.199, 3), (0.20, 4), (0.299, 4), (0.30, 5), (2.0, 5)]
        for value, level in cases:
            with self.subTest(value=value):
                self.assertEqual(psi_level(value, THRESHOLDS), level)

    def test_line_chart_skips_missing_points_and_exposes_values_to_keyboard(self):
        svg = line_chart(["jan/26", "fev/26", "mar/26"], [("AUC", [0.85, None, 0.70], "var(--series-1)")],
                         fmt=num, title="AUC <por> lote", refs=[(0.857, "baseline", "end")])
        self.assertEqual(svg.count('class="dot"'), 2)
        self.assertIn('aria-label="AUC &lt;por&gt; lote"', svg)
        points = json.loads(html.unescape(re.search(r'data-points="([^"]*)"', svg).group(1)))
        self.assertEqual([p["t"] for p in points], ["jan/26", "fev/26", "mar/26"])
        self.assertEqual([[row["v"] for row in p["r"]] for p in points], [["0,850"], [], ["0,700"]])


class BuildTests(unittest.TestCase):
    def test_page_has_every_section_and_the_headline_numbers(self):
        page = build(make_summary(FULL), None, [])
        for title in ["Desempenho e drift ao longo do tempo", "Saúde do pipeline", "Drift por variável e por lote",
                      "Desempenho do modelo", "Alertas", "Atribuição causal", "Definição das métricas"]:
            self.assertIn(f"<h2>{title}", page)
        self.assertIn("3 de 4", page)                                # lotes processados
        self.assertIn("13.236", page)                                # linhas processadas
        self.assertIn("4.612 bloqueadas", page)
        self.assertIn("mai/26".replace("mai", "mar"), page)          # último lote: mar/26
        self.assertIn("alterada na simulação", page)

    def test_signal_board_summarises_each_rule_group_per_batch(self):
        page = build(make_summary(FULL), None, [])
        board = page[page.index('<table class="board"'):page.index("</table>", page.index('<table class="board"'))]
        self.assertEqual(board.count("<td "), 5 * 4)                 # cinco grupos de regras, quatro lotes
        self.assertEqual(board.count('class="s-critical"'), 1)       # a entrega corrompida, no contrato
        self.assertEqual(board.count('class="s-na"'), 3)             # drift e desempenho não avaliados nela
        self.assertEqual(board.count('class="s-alert"'), 3)          # AUC em fev e mar, drift em mar
        self.assertIn("bloqueado", board)
        self.assertIn("<small>entrega corrompida</small>", board)
        self.assertIn("Queda de AUC", board)                         # a mensagem vai no tooltip da célula
        self.assertIn("data-grid", board)
        self.assertNotIn("tabindex", board)                          # o script dá uma parada de Tab por grade
        self.assertEqual(board.count("aria-label="), 5 * 4)          # cada célula tem nome próprio

    def test_rail_lists_run_facts_and_only_existing_sections(self):
        page = build(make_summary(FULL), None, [])
        rail = page[page.index('<aside class="rail"'):page.index("</aside>")]
        for expected in ["mon-teste", "14/11/2026 23:41 UTC", "abc123", "65.867 linhas de treino",
                         'href="#visao"', 'href="#drift"', 'href="#metricas"']:
            self.assertIn(expected, rail)
        self.assertNotIn('href="#cenarios"', rail)                   # sem resumo de cenários, sem item no índice
        self.assertIn('<a class="skip" href="#conteudo">', page)
        self.assertIn('<main class="main" id="conteudo">', page)
        for label in ["Alertas gerados nesta execução", "Status, volume e duração por lote e por etapa",
                      "Definição das métricas e regras de alerta", "Dados do gráfico"]:
            self.assertIn(f'aria-label="{label}"', page)             # toda tabela de dados tem nome
        only_rejected = build(make_summary([rejected_batch()]), None, [])
        self.assertNotIn('href="#drift"', only_rejected)
        self.assertIn('href="#pipeline"', only_rejected)

    def test_page_is_self_contained(self):
        page = build(make_summary(FULL), None, [])
        self.assertTrue(page.startswith("<!DOCTYPE html>"))
        for external in ("http://", "https://", "<link", " src="):
            self.assertNotIn(external, page)

    def test_silent_degradation_is_read_from_the_data(self):
        page = build(make_summary(FULL), None, [])
        self.assertIn("A AUC cruzou o limite de queda (0,05) em <strong>fev/26</strong>", page)
        self.assertIn("1º alerta de AUC", page)
        self.assertIn("1º aviso de PSI", page)                       # mar/26 já entra direto em alerta
        self.assertNotIn("1º alerta de PSI", page)

    def test_one_table_row_per_alert_with_severity_filters(self):
        page = build(make_summary(FULL), None, [])
        self.assertEqual(page.count("<tr data-sev="), 4)
        self.assertEqual(page.count('<tr data-sev="critical"'), 1)
        self.assertIn(">Todos (4)<", page)
        self.assertIn(">Alerta (3)<", page)
        self.assertIn("1 crítico", page)
        self.assertNotIn("1 críticos", page)

    def test_untrusted_text_is_escaped(self):
        batches = [good_batch("2026-01", 0.85, 0.3, [{**drift_alert("2026-01"),
                                                     "message": "<script>alert('x')</script>"}])]
        summary = make_summary(batches)
        summary["run_id"] = '"><img onerror=alert(1)>'
        page = build(summary, None, [])
        self.assertNotIn("<script>alert", page)
        self.assertNotIn("<img onerror", page)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", page)

    def test_run_without_successful_batches_still_renders(self):
        page = build(make_summary([rejected_batch()]), None, [])
        self.assertIn("Nenhum lote foi processado com sucesso", page)
        self.assertIn("0 de 1", page)
        self.assertNotIn("Drift por variável e por lote", page)
        self.assertIn("valor nulo (3 linhas)", page)

    def test_pipeline_health_prefers_the_central_event_log(self):
        summary = make_summary(FULL)
        events = [{"ts": "2026-11-14T23:41:33.177+00:00", "run_id": "mon-teste", "stage": "drift", "batch": "2026-01",
                   "status": "success", "level": "INFO", "duration_s": 7.77, "rows": 4412, "details": {}},
                  {"ts": "2026-11-14T23:41:33.177+00:00", "run_id": "outra-execucao", "stage": "drift",
                   "batch": "2026-02", "status": "failed", "level": "ERROR", "duration_s": 9.99, "rows": 1, "details": {}},
                  {"ts": "2026-11-14T23:40:00.000+00:00", "run_id": "train-1", "stage": "train", "batch": None,
                   "status": "success", "level": "INFO", "duration_s": 3.21, "rows": None, "details": {}}]
        page = build(summary, None, events)
        self.assertIn("7,77 s", page)                                 # evento desta execução
        self.assertNotIn("9,99 s", page)                              # evento de outra execução é ignorado
        self.assertIn("Treino do baseline", page)
        self.assertIn("1 evento nesta execução", page)
        self.assertIn("Log central indisponível", build(summary, None, []))

    def test_causal_section_shows_scenarios_or_a_placeholder(self):
        self.assertIn("Análise causal ainda não executada", build(make_summary(FULL), None, []))
        metrics = {"auc": 0.8, "ks": 0.5, "recall": 0.7, "precision": 0.2, "default_rate": 0.06,
                   "mean_score": 0.3, "flagged_rate": 0.24}
        effect = {key: -0.064 for key in metrics}
        scenario = lambda ci: {"metrics": metrics, "effect": effect, "auc_effect_ci95": ci, "psi_score": 0.39}
        causal = {"rows": 4412, "month": "2026-03",
                  "scenarios": {"controle": scenario(None), "data_drift": scenario([-0.078, -0.047]),
                                "concept_drift": scenario([-0.219, -0.164]), "ambos": scenario([-0.160, -0.115])},
                  "feature_interventions": {"MonthlyIncome": scenario([-0.01, 0.0])},
                  "auc_decomposition": {"data_drift": -0.064, "concept_drift": -0.189, "interaction": 0.114, "total": -0.138}}
        page = build(make_summary(FULL), causal, [])
        for expected in ["Só P(X): data drift", "−0,189", "+0,114", "[−0,078; −0,047]", "Renda mensal"]:
            self.assertIn(expected, page)


    def test_paired_scenarios_section_is_optional(self):
        self.assertNotIn("Cenários pareados da Etapa 2", build(make_summary(FULL), None, []))
        drift = lambda income, score: {
            column: {"psi": value, "status": status_for(value)}
            for column, value in {"MonthlyIncome": income, "DebtRatio": income / 2, config.SCORE: score}.items()}
        metrics = {"auc": 0.843, "recall": 0.74, "flagged_rate": 0.25}
        scenarios = {"note": "Rótulos <preservados>.", "scenarios": [
            {"scenario": "t0_control", "rows": 35297, "factors": {"MonthlyIncome": 1.0, "DebtRatio": 1.0},
             "report": "reports/drift/t0_control.html", "evidently_drifted": [], "drift": drift(0.0, 0.001),
             "performance": metrics},
            {"scenario": "t3", "rows": 35297, "factors": {"MonthlyIncome": 0.6, "DebtRatio": 1.6},
             "report": "reports/drift/t3.html", "evidently_drifted": ["MonthlyIncome", "DebtRatio"],
             "drift": drift(0.6, 0.034), "performance": metrics}]}
        page = build(make_summary(FULL), None, [], scenarios)
        for expected in ['id="cenarios"', "Cenários pareados da Etapa 2", "×0,60", "×1,60", "▲ 0,600",
                         "o PSI da renda chega a 0,600", "Rótulos &lt;preservados&gt;.", "35.297 clientes",
                         "<td>nenhuma</td>", "<td>2 de 2</td>"]:
            self.assertIn(expected, page)
        only_rejected = build(make_summary([rejected_batch()]), None, [], scenarios)
        self.assertIn("Cenários pareados da Etapa 2", only_rejected)


class MainTests(TempDirTestCase):
    def test_missing_summary_returns_two_and_writes_nothing(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main([]), 2)
        self.assertIn("src.monitor", stderr.getvalue())
        self.assertFalse(config.DASHBOARD.exists())

    def test_writes_the_dashboard_from_the_reports(self):
        config.SUMMARY.parent.mkdir(parents=True)
        config.SUMMARY.write_text(json.dumps(make_summary(FULL)), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        page = config.DASHBOARD.read_text(encoding="utf-8")
        self.assertIn("Drift, desempenho e saúde do pipeline", page)
        self.assertIn("evento", page)                                # o próprio dashboard registra seu evento


if __name__ == "__main__":
    unittest.main()
