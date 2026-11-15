import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src import config
from src.ingest import CONTRACT_REJECTED, OK, summarize_violations, validate_batch
from src.observability import log_event, new_run_id, read_events, track
from support import TempDirTestCase, synthetic_credit

EVENT_KEYS = {"ts", "run_id", "stage", "batch", "status", "level", "duration_s", "rows", "details"}


class EventLogTests(TempDirTestCase):
    def test_event_is_appended_with_the_documented_schema(self):
        run_id = new_run_id("teste")
        log_event("ingest", "success", batch="2026-01", rows=np.int64(10), duration_s=np.float64(0.12345678),
                  path=Path("data/lote.parquet"), share=np.float32(0.5))
        log_event("drift", "success", batch="2026-01")
        first, second = read_events()
        self.assertEqual(set(first), EVENT_KEYS)
        self.assertEqual((first["run_id"], first["stage"], first["batch"], first["status"], first["level"]),
                         (run_id, "ingest", "2026-01", "success", "INFO"))
        self.assertEqual((first["rows"], first["duration_s"]), (10, 0.1235))
        self.assertEqual(first["details"], {"path": "data/lote.parquet", "share": 0.5})
        self.assertEqual((second["rows"], second["duration_s"], second["details"]), (None, None, {}))
        lines = config.EVENT_LOG.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line) for line in lines], [first, second])       # uma linha por evento

    def test_run_ids_are_unique_and_explicit_run_id_wins(self):
        first, second = new_run_id("mon"), new_run_id("mon")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("mon-"))
        self.assertEqual(log_event("score", "success")["run_id"], second)
        self.assertEqual(log_event("score", "success", run_id="outra")["run_id"], "outra")

    def test_track_records_success_with_duration_rows_and_details(self):
        with track("score", batch="2026-02") as info:
            info.update(rows=42, report="relatorio.html")
        event = read_events()[-1]
        self.assertEqual((event["stage"], event["batch"], event["status"], event["rows"]),
                         ("score", "2026-02", "success", 42))
        self.assertGreaterEqual(event["duration_s"], 0)
        self.assertEqual(event["details"], {"report": "relatorio.html"})
        self.assertAlmostEqual(info["duration_s"], event["duration_s"], places=3)

    def test_track_records_failure_and_reraises(self):
        with self.assertRaises(KeyError):
            with track("drift", batch="2026-03"):
                raise KeyError("coluna ausente")
        event = read_events()[-1]
        self.assertEqual((event["stage"], event["batch"], event["status"], event["level"]),
                         ("drift", "2026-03", "failed", "ERROR"))
        self.assertIn("KeyError", event["details"]["error"])

    def test_unwritable_log_never_breaks_the_pipeline(self):
        config.LOG_DIR.write_text("um arquivo no lugar do diretório de logs")
        with self.assertLogs("pipeline", level="WARNING"):
            event = log_event("ingest", "success", rows=1)
        self.assertEqual(event["status"], "success")
        with self.assertLogs("pipeline", level="WARNING"):
            log_event("ingest", "success", payload=object())          # detalhe não serializável

    def test_read_events_skips_truncated_lines_and_missing_file(self):
        self.assertEqual(read_events(), [])
        log_event("ingest", "success")
        with config.EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write('{"ts": "2026-01-01T00:00:00", "stage": "dri')   # escrita interrompida
        self.assertEqual([event["stage"] for event in read_events()], ["ingest"])


class IngestSummaryTests(TempDirTestCase):
    def write(self, df, name="lote.parquet"):
        path = Path("data/incoming") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return path

    def test_violations_are_aggregated_by_rule_without_values(self):
        cases = pd.DataFrame({
            "column": ["age", "age", "MonthlyIncome", "age", "DebtRatio", None],
            "check": ["in_range(18, 100)", "in_range(18, 100)", "not_nullable",
                      "multiple_fields_uniqueness", "multiple_fields_uniqueness", "non_empty"],
            "failure_case": [15, 0, None, 45, 0.5, False],
            "index": [0, 1, 2, 3, 3, None],
        })
        summary = summarize_violations(cases)
        self.assertEqual(summary[0], {"column": "age", "check": "in_range(18, 100)", "rows": 2})
        self.assertIn({"column": None, "check": "linhas duplicadas", "rows": 1}, summary)     # uma vez, não por coluna
        self.assertIn({"column": "MonthlyIncome", "check": "not_nullable", "rows": 1}, summary)
        self.assertIn({"column": None, "check": "non_empty", "rows": 1}, summary)
        self.assertEqual(len(summary), 4)
        for item in summary:
            self.assertEqual(set(item), {"column", "check", "rows"})

    def test_valid_batch_is_approved_and_logged(self):
        result = validate_batch(self.write(synthetic_credit(200, seed=4)))
        self.assertEqual((result.code, result.rows, result.violations, result.quarantine), (OK, 200, [], None))
        event = read_events()[-1]
        self.assertEqual((event["stage"], event["batch"], event["status"], event["rows"]),
                         ("ingest", "lote", "success", 200))

    def test_rejected_batch_goes_to_the_given_quarantine_and_log_has_no_values(self):
        df = synthetic_credit(200, seed=4)
        df.loc[:9, "age"] = 15
        df.loc[10:14, "MonthlyIncome"] = np.nan
        secret = 987654321.0                                          # valor rejeitado que não pode vazar
        df.loc[20, "DebtRatio"] = -secret
        path = self.write(df)
        with self.assertLogs("ingest", level="ERROR"):
            result = validate_batch(path, "data/quarantine/run-1")
        self.assertEqual(result.code, CONTRACT_REJECTED)
        self.assertEqual(result.quarantine, "data/quarantine/run-1/lote.parquet")
        self.assertFalse(path.exists())
        self.assertTrue(Path(result.quarantine).exists())
        self.assertEqual({(v["column"], v["rows"]) for v in result.violations},
                         {("age", 10), ("MonthlyIncome", 5), ("DebtRatio", 1)})
        raw_log = config.EVENT_LOG.read_text(encoding="utf-8")
        self.assertNotIn("987654321", raw_log)                        # agregados, nunca valores
        self.assertIn("987654321", Path("data/quarantine/run-1/lote_errors.csv").read_text())
        event = read_events()[-1]
        self.assertEqual((event["status"], event["level"], event["rows"]), ("rejected", "ERROR", 200))
        self.assertEqual(event["details"]["violations"], result.violations)


if __name__ == "__main__":
    unittest.main()
