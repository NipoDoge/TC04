import json
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.demo_observability import demonstrate
from src.execution_log import append_event, record, tracked
from src.ingest import ingest
from src.observability import STAGES, collect, read_executions, run
from support import synthetic_credit


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = self.root / "executions.jsonl"
        env = patch.dict(os.environ, {"TC04_EXECUTION_LOG": str(self.log), "TC04_RUN_ID": "test-run"})
        env.start()
        self.addCleanup(env.stop)

    def events(self):
        events, invalid = read_executions(self.log)
        self.assertFalse(invalid)
        return events

    def test_real_duration_resources_and_correlated_start_finish(self):
        @tracked("prepare")
        def execute():
            record(volume=12, input_rows=15)
            return "result"
        self.assertEqual(execute(), "result")
        start, finish = self.events()
        self.assertEqual(start["execution_id"], finish["execution_id"])
        self.assertEqual(finish["run_id"], "test-run")
        self.assertGreater(finish["duration_ms"], 0)
        self.assertEqual(finish["volume"], 12)
        self.assertGreater(finish["infrastructure"]["disk_total_bytes"], 0)
        self.assertGreater(finish["infrastructure"]["process_peak_rss_mb"], 0)

    def test_failure_preserves_exception_without_customer_values(self):
        @tracked("prepare")
        def execute():
            raise ValueError("private customer value 12345")
        with self.assertRaisesRegex(ValueError, "private customer"):
            execute()
        event = self.events()[-1]
        self.assertEqual(event["status"], "fail")
        self.assertEqual(event["error_type"], "ValueError")
        self.assertNotIn("12345", self.log.read_text())

    def test_last_success_recovers_despite_historical_rejection_and_quarantine(self):
        @tracked("ingest", exit_codes=True)
        def execute(code):
            record(volume=500)
            return code
        for code in (2, 1, 0):
            execute(code)
        q = self.root / "quarantine"
        q.mkdir()
        (q / "bad.parquet").write_bytes(b"historical")
        (q / "bad_errors.csv").write_text("historical")
        m = collect(quarantine_dir=q)["metrics"]
        stage = next(s for s in m["pipeline"]["stages"] if s["name"] == "ingest")
        self.assertEqual(stage["status"], "ok")
        self.assertEqual(stage["volume"], 500)
        self.assertEqual(m["pipeline"]["execution_counts"]["fail"], 1)
        self.assertEqual(m["contract_alerts"]["rejected_execution_count"], 1)
        self.assertEqual(m["contract_alerts"]["quarantined_batch_count"], 1)

    def test_real_ingest_read_failure_is_logged(self):
        with self.assertLogs("ingest", level="ERROR"):
            self.assertEqual(ingest(self.root / "missing.parquet"), 2)
        event = self.events()[-1]
        self.assertEqual(event["status"], "fail")
        self.assertEqual(event["error_type"], "FileNotFoundError")
        self.assertIsNone(event["volume"])

    def test_unknown_stages_are_not_inferred_from_historical_artifacts(self):
        metrics = collect()["metrics"]
        self.assertTrue(all(s["status"] == "unknown" for s in metrics["pipeline"]["stages"]))
        self.assertEqual(metrics["pipeline"]["status"], "incomplete")

    def test_malformed_lines_mark_collector_and_pipeline_failed(self):
        self.log.write_text('{"unfinished":\n[]\n')
        metrics = collect()["metrics"]
        self.assertEqual(metrics["collector"]["invalid_lines"], [1, 2])
        self.assertEqual(metrics["pipeline"]["status"], "fail")

    def test_unfinished_execution_becomes_stale(self):
        timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        append_event({"execution_id": "interrupted", "run_id": "run", "ts": timestamp,
                      "started_at_utc": timestamp, "event": "execution_started", "source": "predict", "status": "running"})
        metrics = collect()["metrics"]
        self.assertEqual(next(s for s in metrics["pipeline"]["stages"] if s["name"] == "predict")["status"], "stale")
        self.assertEqual(metrics["pipeline"]["status"], "fail")

    def test_completed_execution_expiry_and_run_filter(self):
        @tracked("prepare")
        def execute():
            record(volume=3)
        execute()
        future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        metrics = collect(ts=future)["metrics"]
        self.assertEqual(metrics["pipeline"]["stages"][0]["status"], "stale")
        self.assertEqual(collect(run_id="other")["metrics"]["pipeline"]["execution_count"], 0)

    def test_collect_preserves_source_journal_and_escapes_history(self):
        @tracked("ingest", exit_codes=True)
        def execute():
            return 0
        with patch.dict(os.environ, {"TC04_RUN_ID": '<script>alert("x")</script>'}):
            execute()
        before = self.log.read_bytes()
        _, paths = run(output_dir=self.root / "dashboard")
        run(output_dir=self.root / "dashboard")
        self.assertEqual(self.log.read_bytes(), before)
        html = paths["index_path"].read_text()
        self.assertNotIn('<script>alert("x")</script>', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("event-filter", html)

    def test_logging_failure_prevents_unobserved_operation(self):
        called = []
        @tracked("prepare")
        def execute():
            called.append(True)
        with patch("src.execution_log.append_event", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                execute()
        self.assertFalse(called)

    def test_parallel_writers_preserve_complete_json_lines(self):
        @tracked("predict")
        def execute(number):
            record(volume=number)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(execute, range(1, 21)))
        events = self.events()
        self.assertEqual(len(events), 40)
        self.assertEqual(len({e["execution_id"] for e in events}), 20)
        self.assertEqual(sum(e["event"] == "execution_finished" for e in events), 20)

    def test_invalid_model_and_drift_artifacts_fail_explicitly(self):
        invalid = self.root / "invalid.json"
        for data in ([], None, {"auc": None}, {"scenarios": [None]}):
            invalid.write_text(json.dumps(data))
            metrics = collect(model_metrics=invalid, drift_manifest=invalid, concept_manifest=invalid)["metrics"]
            self.assertEqual(metrics["model"]["status"], "fail")
            self.assertEqual(metrics["drift"]["status"], "fail")
            self.assertEqual(metrics["performance"]["status"], "fail")
            self.assertEqual(metrics["pipeline"]["status"], "fail")

    def test_full_demo_runs_real_pipeline_and_retains_incidents(self):
        raw = self.root / "raw.csv"
        synthetic_credit().to_csv(raw, index=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertLogs("ingest", level="ERROR"):
            summary = demonstrate(raw, self.root / "demo", self.log)
        pipeline = summary["pipeline"]
        self.assertEqual(pipeline["execution_count"], 10)
        self.assertEqual(pipeline["execution_counts"]["fail"], 2)
        self.assertEqual(pipeline["execution_counts"]["rejected"], 1)
        self.assertEqual(next(s for s in pipeline["stages"] if s["name"] == "ingest")["status"], "ok")
        self.assertTrue(all(s["duration_ms"] is not None and s["volume"] > 0 for s in pipeline["stages"]))
        self.assertEqual(pipeline["status"], "alert")
        metrics = json.loads((self.root / "demo/dashboard/metrics.json").read_text())
        self.assertEqual(metrics["performance"]["alert_count"], 2)
        self.assertGreater(metrics["drift"]["alerts"]["dataset_drift_count"], 0)
        self.assertEqual(len(self.events()), 20)
        # Adaptado de test_monitor.py::test_central_log_has_one_event_per_stage_and_alert (4f50a83).
        events = self.events()
        self.assertEqual({e["run_id"] for e in events}, {summary["run_id"]})
        finished = [e for e in events if e["event"] == "execution_finished"]
        self.assertEqual({e["source"] for e in finished}, set(STAGES))
        for execution_id in {e["execution_id"] for e in events}:
            pair = [e for e in events if e["execution_id"] == execution_id]
            self.assertEqual([e["event"] for e in pair], ["execution_started", "execution_finished"])
            self.assertEqual(pair[0]["source"], pair[1]["source"])
        for source, alerts in (("detect_drift", metrics["drift"]["alerts"]["dataset_drift_count"]),
                               ("concept_drift", metrics["performance"]["alert_count"])):
            self.assertEqual(next(e for e in finished if e["source"] == source)["alert_count"], alerts)


if __name__ == "__main__":
    unittest.main()
