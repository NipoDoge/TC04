"""Casos adaptados de nipodoge/feat/fase4-entrega-final, commit 4f50a83.

Origens: tests/test_observability.py, test_dashboard.py, test_alerts.py e
 test_monitor.py. As APIs e os estados seguem a implementação atual.
"""
import contextlib
import io
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.concept_drift import classification_metrics, performance_degraded
from src.config import CONCEPT_AUC_DROP, CONCEPT_RECALL_DROP
from src.execution_log import append_event, record, tracked
from src.ingest import ingest, summarize_violations
from src.detect_drift import detect_drift
from src.observability import collect, main, read_executions, render_html, run
from src.simulate_drift import simulate
from support import synthetic_credit


class TemporaryObservability(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.journal = self.root / 'executions.jsonl'
        environment = patch.dict(os.environ, TC04_EXECUTION_LOG=str(self.journal), TC04_RUN_ID='remote-tests')
        environment.start()
        self.addCleanup(environment.stop)

    def events(self):
        events, invalid = read_executions(self.journal)
        self.assertEqual(invalid, [])
        return events

    def execute(self, source='ingest', code=0, rows=200):
        @tracked(source, exit_codes=True)
        def operation():
            record(volume=rows)
            return code
        return operation()

    def sources(self):
        """Fontes agregadas mínimas; nenhuma dependência do dataset versionado."""
        model = self.root / 'model.json'
        model.write_text(json.dumps({'auc': .85, 'recall_at_0.5': .77, 'ks': .56}))
        drift = self.root / 'drift/manifest.json'
        drift.parent.mkdir(exist_ok=True)
        drift.write_text(json.dumps({'scenarios': [
            {'name': 't0_control', 'rows': 200, 'drift_share': 0., 'dataset_drift_detected': False, 'drifted_features': []},
            {'name': 't3', 'rows': 200, 'drift_share': .2, 'dataset_drift_detected': True, 'drifted_features': ['age']},
        ]}))
        for name in ['t0_control', 't3']:
            (drift.parent / f'{name}.html').write_text('<html>report</html>')
        concept = self.root / 'concept/manifest.json'
        concept.parent.mkdir(exist_ok=True)
        concept.write_text(json.dumps({'results': [
            {'scenario': 'c0_control', 'auc': .85, 'recall': .77, 'auc_drop': 0., 'recall_drop': 0., 'performance_alert': False},
            {'scenario': 'c3', 'auc': .65, 'recall': .47, 'auc_drop': .2, 'recall_drop': .3, 'performance_alert': True},
        ]}))
        (concept.parent / 'analysis.html').write_text('<html>analysis</html>')
        scenarios = self.root / 'scenarios.json'
        scenarios.write_text(json.dumps({'periods': [{'name': 't0_control', 'rows': 200}]}))
        predictions = self.root / 'predictions.json'
        predictions.write_text(json.dumps({'datasets': [{'name': 'reference', 'rows': 200}]}))
        return dict(execution_log=self.journal, model_metrics=model, drift_manifest=drift,
                    concept_manifest=concept, scenarios_manifest=scenarios, predictions_manifest=predictions,
                    quarantine_dir=self.root / 'quarantine')


class RemoteEventLogTests(TemporaryObservability):
    def test_event_is_appended_with_schema_and_numpy_path_serialization(self):
        @tracked('predict')
        def operation():
            record(volume=np.int64(10), report=Path('report.html'), share=np.float32(.5))
        operation()
        start, finish = self.events()
        self.assertEqual(set(start), {'schema_version', 'run_id', 'execution_id', 'source', 'started_at_utc', 'ts', 'event', 'status'})
        self.assertEqual((finish['run_id'], finish['source'], finish['status']), ('remote-tests', 'predict', 'ok'))
        self.assertEqual((finish['volume'], finish['report'], finish['share']), (10, 'report.html', .5))
        self.assertEqual([json.loads(line) for line in self.journal.read_text().splitlines()], [start, finish])

    def test_execution_ids_are_unique_and_explicit_run_id_wins(self):
        self.execute()
        self.execute()
        with patch.dict(os.environ, TC04_RUN_ID='other-run'):
            self.execute()
        finishes = [e for e in self.events() if e['event'] == 'execution_finished']
        self.assertEqual(len({e['execution_id'] for e in finishes}), 3)
        self.assertEqual([e['run_id'] for e in finishes], ['remote-tests', 'remote-tests', 'other-run'])

    def test_track_records_success_with_duration_rows_and_details(self):
        self.execute('predict', rows=42)
        event = self.events()[-1]
        self.assertEqual((event['source'], event['status'], event['volume']), ('predict', 'ok', 42))
        self.assertGreaterEqual(event['duration_ms'], 0)
        self.assertEqual(event['volume_unit'], 'rows')
        self.assertIsNone(event['error_type'])

    def test_track_records_failure_and_reraises(self):
        @tracked('detect_drift')
        def operation():
            raise KeyError('missing column')
        with self.assertRaises(KeyError):
            operation()
        event = self.events()[-1]
        self.assertEqual((event['source'], event['status'], event['error_type']), ('detect_drift', 'fail', 'KeyError'))
        self.assertGreaterEqual(event['duration_ms'], 0)

    def test_read_events_preserves_valid_events_before_truncated_line(self):
        self.assertEqual(read_executions(self.journal), ([], []))
        self.execute()
        before = self.events()
        with self.journal.open('a') as stream:
            stream.write('{"ts":"unfinished')
        events, invalid = read_executions(self.journal)
        self.assertEqual(events, before)
        self.assertEqual(invalid, [3])
        self.assertEqual(collect(execution_log=self.journal)['metrics']['collector']['status'], 'fail')


class RemoteIngestSummaryTests(TemporaryObservability):
    def test_violations_are_aggregated_by_rule_without_values(self):
        cases = pd.DataFrame({
            'column': ['age', 'age', 'MonthlyIncome', 'age', 'DebtRatio', None],
            'check': ['in_range(18, 100)', 'in_range(18, 100)', 'not_nullable',
                      'multiple_fields_uniqueness', 'multiple_fields_uniqueness', 'non_empty'],
            'failure_case': [15, 0, None, 45, .5, False], 'index': [0, 1, 2, 3, 3, None],
        })
        summary = summarize_violations(cases)
        self.assertEqual(summary[0], {'column': 'age', 'check': 'in_range(18, 100)', 'rows': 2})
        self.assertIn({'column': None, 'check': 'linhas duplicadas', 'rows': 1}, summary)
        self.assertIn({'column': 'MonthlyIncome', 'check': 'not_nullable', 'rows': 1}, summary)
        self.assertIn({'column': None, 'check': 'non_empty', 'rows': 1}, summary)
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(set(item) == {'column', 'check', 'rows'} for item in summary))

    def test_valid_batch_is_approved_and_logged(self):
        path = self.root / 'good.parquet'
        synthetic_credit(200, seed=4).to_parquet(path)
        self.assertEqual(ingest(path), 0)
        event = self.events()[-1]
        self.assertEqual((event['source'], event['status'], event['volume']), ('ingest', 'ok', 200))
        self.assertTrue(path.exists())

    def test_rejected_batch_uses_given_quarantine_and_no_values_leak(self):
        data = synthetic_credit(200, seed=4)
        data.loc[:9, 'age'] = 15
        data.loc[10:14, 'MonthlyIncome'] = np.nan
        data.loc[20, 'DebtRatio'] = -987654321.
        path = self.root / 'bad.parquet'
        data.to_parquet(path)
        quarantine = self.root / 'quarantine/run-1'
        with self.assertLogs('ingest', level='ERROR'):
            self.assertEqual(ingest(path, quarantine), 1)
        event = self.events()[-1]
        self.assertEqual((event['status'], event['volume']), ('rejected', 200))
        self.assertEqual({(v['column'], v['rows']) for v in event['violations']},
                         {('age', 10), ('MonthlyIncome', 5), ('DebtRatio', 1)})
        self.assertFalse(path.exists())
        self.assertTrue(Path(event['quarantine']).exists())
        self.assertNotIn('987654321', self.journal.read_text())
        self.assertNotIn('failure_case', self.journal.read_text())
        self.assertIn('987654321', (quarantine / 'bad_errors.csv').read_text())


class RemoteDashboardTests(TemporaryObservability):
    def test_data_tables_have_accessible_names(self):
        payload = collect(**self.sources())
        class Tables(HTMLParser):
            def __init__(self):
                super().__init__(); self.names = []
            def handle_starttag(self, tag, attrs):
                if tag == 'table': self.names.append(dict(attrs).get('aria-label'))
        parser = Tables()
        parser.feed(render_html(payload['metrics'], payload['events']))
        self.assertEqual(len(parser.names), 7)
        self.assertTrue(all(parser.names))
        self.assertEqual(len(set(parser.names)), 7)

    def test_page_has_sections_and_headline_numbers(self):
        self.execute(rows=4412)
        payload = collect(**self.sources())
        page = render_html(payload['metrics'], payload['events'])
        for title in ('Últimas execuções reais', 'Infraestrutura no momento da coleta',
                      'Desempenho sob concept drift', 'Drift por cenário', 'Histórico de eventos reais'):
            self.assertIn(f'<h2>{title}', page)
        self.assertIn('<td>4412</td>', page)
        self.assertIn('drift de dados = 1', page)
        self.assertIn('alertas de desempenho = 1', page)
        self.assertIn('remote-tests', page)

    def test_page_is_self_contained(self):
        payload = collect(**self.sources())
        page = render_html(payload['metrics'], payload['events'])
        self.assertTrue(page.startswith('<!DOCTYPE html>'))
        for external in ('http://', 'https://', '<link', ' src='):
            self.assertNotIn(external, page)

    def test_pipeline_health_prefers_central_log_and_filters_other_runs(self):
        self.execute('predict', rows=4412)
        with patch.dict(os.environ, TC04_RUN_ID='other-run'):
            self.execute('predict', code=2, rows=1)
        payload = collect(**self.sources(), run_id='remote-tests')
        stage = next(s for s in payload['metrics']['pipeline']['stages'] if s['name'] == 'predict')
        self.assertEqual((stage['status'], stage['volume']), ('ok', 4412))
        page = render_html(payload['metrics'], payload['events'])
        self.assertNotIn('other-run', page)
        self.assertEqual(payload['metrics']['pipeline']['execution_count'], 1)

    def test_untrusted_text_is_escaped_everywhere(self):
        with patch.dict(os.environ, TC04_RUN_ID='\"><img onerror=alert(1)>'):
            self.execute()
        sources = self.sources()
        manifest = json.loads(sources['drift_manifest'].read_text())
        manifest['scenarios'][-1]['drifted_features'] = ["<script>alert('x')</script>"]
        sources['drift_manifest'].write_text(json.dumps(manifest))
        payload = collect(**sources)
        page = render_html(payload['metrics'], payload['events'])
        self.assertNotIn('<img onerror', page)
        self.assertNotIn('<script>alert', page)
        self.assertIn('&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;', page)

    def test_run_without_successful_executions_still_renders(self):
        self.execute(code=1)
        payload, paths = run(output_dir=self.root / 'dashboard', **self.sources())
        self.assertEqual(payload['metrics']['pipeline']['status'], 'fail')
        self.assertIn('rejected', paths['index_path'].read_text())
        self.assertEqual(payload['metrics']['pipeline']['execution_counts']['ok'], 0)

    def test_missing_sources_write_diagnostic_dashboard_without_false_success(self):
        missing = self.root / 'missing.json'
        sources = {key: missing for key in self.sources() if key != 'quarantine_dir'}
        payload, paths = run(output_dir=self.root / 'dashboard', **sources)
        self.assertEqual(payload['metrics']['pipeline']['status'], 'incomplete')
        self.assertTrue(paths['index_path'].exists())
        self.assertTrue(all(s['status'] == 'unknown' for s in payload['metrics']['pipeline']['stages']))

    def test_report_links_resolve_from_arbitrary_dashboard_directory(self):
        _, paths = run(output_dir=self.root / 'custom/nested/dashboard', **self.sources())
        class Links(HTMLParser):
            def __init__(self):
                super().__init__(); self.links = []
            def handle_starttag(self, tag, attrs):
                if tag == 'a': self.links.append(dict(attrs)['href'])
        parser = Links()
        parser.feed(paths['index_path'].read_text())
        self.assertEqual(len(parser.links), 3)
        self.assertTrue(all((paths['index_path'].parent / link).is_file() for link in parser.links))

    def test_main_writes_dashboard_from_selected_sources(self):
        arguments = ['--output-dir', str(self.root / 'dashboard')]
        for key, path in self.sources().items():
            arguments.extend(['--' + key.replace('_', '-'), str(path)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        self.assertIn('Observabilidade do pipeline', (self.root / 'dashboard/index.html').read_text())


class ObservabilityRegressionTests(TemporaryObservability):
    def test_real_drift_failure_is_logged_without_partial_report_or_customer_values(self):
        # Adaptado de test_monitor.py::test_monitor_survives_without_mlflow_and_reports_operational_failure.
        reference = self.root / 'reference.parquet'
        synthetic_credit(200).to_parquet(reference)
        scenarios = self.root / 'scenarios'
        simulate(reference, scenarios)
        inputs = [reference, *scenarios.glob('*')]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        output = self.root / 'failed-reports'
        with patch('src.detect_drift.compare_datasets', side_effect=RuntimeError('private value 987654321')):
            with self.assertRaises(RuntimeError):
                detect_drift(reference, scenarios, output)
        event = self.events()[-1]
        self.assertEqual((event['source'], event['status'], event['error_type']), ('detect_drift', 'fail', 'RuntimeError'))
        self.assertFalse(output.exists())
        self.assertNotIn('987654321', self.journal.read_text())
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, before)

    def test_final_log_failure_does_not_mask_original_operation_error(self):
        @tracked('prepare')
        def operation():
            raise ValueError('original operation error')
        with patch('src.execution_log.append_event', side_effect=[None, OSError('disk failed')]):
            with self.assertLogs('pipeline', level='ERROR'), self.assertRaisesRegex(ValueError, 'original operation'):
                operation()

    def test_latest_execution_compares_instants_across_timezone_offsets(self):
        for identifier, timestamp, code in [('older', '2026-11-17T12:00:00+02:00', 2),
                                             ('newer', '2026-11-17T10:30:00+00:00', 0)]:
            with patch('src.execution_log.now', return_value=timestamp), patch.dict(os.environ, TC04_RUN_ID=identifier):
                self.execute(code=code)
        metrics = collect(execution_log=self.journal, ts='2026-11-17T11:00:00+00:00')['metrics']
        stage = next(s for s in metrics['pipeline']['stages'] if s['name'] == 'ingest')
        self.assertEqual((stage['run_id'], stage['status']), ('newer', 'ok'))

    def test_invalid_nested_event_is_flagged_without_crashing(self):
        self.execute()
        event = self.events()[-1]
        event['infrastructure'] = [1]
        self.journal.write_text(json.dumps(event) + '\n')
        payload = collect(execution_log=self.journal)
        self.assertEqual(payload['metrics']['collector']['invalid_lines'], [1])
        self.assertEqual(payload['metrics']['pipeline']['status'], 'fail')

    def test_invalid_drift_feature_does_not_break_rendering(self):
        sources = self.sources()
        manifest = json.loads(sources['drift_manifest'].read_text())
        manifest['scenarios'][0]['drifted_features'] = [None]
        sources['drift_manifest'].write_text(json.dumps(manifest))
        payload, _ = run(output_dir=self.root / 'dashboard', **sources)
        self.assertEqual(payload['metrics']['drift']['status'], 'fail')

    def test_nonfinite_concept_metric_is_reported_as_invalid_source(self):
        sources = self.sources()
        manifest = json.loads(sources['concept_manifest'].read_text())
        manifest['results'][0]['auc'] = float('nan')
        sources['concept_manifest'].write_text(json.dumps(manifest))
        payload, _ = run(output_dir=self.root / 'dashboard', **sources)
        self.assertEqual(payload['metrics']['performance']['status'], 'fail')

    def test_corrupted_prediction_manifest_is_not_reported_as_available(self):
        sources = self.sources()
        sources['predictions_manifest'].write_text('not json')
        metrics = collect(**sources)['metrics']
        self.assertEqual(metrics['pipeline']['status'], 'fail')
        self.assertTrue(metrics['artifacts']['predictions_manifest_exists'])
        self.assertEqual(metrics['artifacts']['predictions_manifest_status'], 'fail')

    def test_failed_render_preserves_all_previous_snapshot_files(self):
        directory = self.root / 'dashboard'
        _, paths = run(output_dir=directory, **self.sources())
        before = {key: path.read_bytes() for key, path in paths.items()}
        self.execute()
        with patch('src.observability.render_html', side_effect=ValueError('render failed')):
            with self.assertRaises(ValueError):
                run(output_dir=directory, **self.sources())
        self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, before)

    def test_snapshot_cannot_replace_execution_journal(self):
        sources = self.sources()
        directory = self.root / 'dashboard'
        directory.mkdir()
        with patch.dict(os.environ, TC04_EXECUTION_LOG=str(directory / 'events.jsonl')):
            self.execute()
        sources['execution_log'] = directory / 'events.jsonl'
        before = sources['execution_log'].read_bytes()
        with self.assertRaises(ValueError):
            run(output_dir=directory, **sources)
        self.assertEqual(sources['execution_log'].read_bytes(), before)

    def test_event_with_nonfinite_detail_is_flagged_without_breaking_snapshot(self):
        self.execute()
        event = self.events()[-1]
        event['cpu_seconds'] = float('nan')
        self.journal.write_text(json.dumps(event) + '\n')
        payload, _ = run(output_dir=self.root / 'dashboard', execution_log=self.journal)
        self.assertEqual(payload['metrics']['collector']['invalid_lines'], [1])

    def test_invalid_utf8_line_does_not_discard_valid_history(self):
        self.execute()
        with self.journal.open('ab') as stream:
            stream.write(b'\xff\n')
        payload, _ = run(output_dir=self.root / 'dashboard', execution_log=self.journal)
        self.assertEqual(payload['metrics']['collector']['invalid_lines'], [3])
        self.assertEqual(len(payload['events']), 2)

    def test_snapshot_staging_failure_preserves_previous_files(self):
        directory = self.root / 'dashboard'
        sources = self.sources()
        _, paths = run(output_dir=directory, **sources)
        before = {key: path.read_bytes() for key, path in paths.items()}
        original = Path.write_text
        def fail_metrics(path, content, **kwargs):
            if path.name == 'metrics.json' and path.parent.name.startswith('.snapshot-'):
                raise OSError('disk full')
            return original(path, content, **kwargs)
        with patch.object(Path, 'write_text', fail_metrics), self.assertRaises(OSError):
            run(output_dir=directory, **sources)
        self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, before)

    def test_final_log_failure_after_success_is_propagated(self):
        with patch('src.execution_log.append_event', side_effect=[None, OSError('disk full')]):
            with self.assertRaises(OSError):
                self.execute()


class RemotePerformanceTests(unittest.TestCase):
    def test_metrics_on_a_scored_batch(self):
        metrics = classification_metrics([0, 0, 1, 1], [.1, .6, .4, .9], [0, 1, 0, 1])
        self.assertAlmostEqual(metrics['auc'], .75)
        self.assertEqual((metrics['recall'], metrics['precision']), (.5, .5))
        self.assertEqual((metrics['prevalence'], metrics['predicted_positive_rate']), (.5, .5))
        self.assertAlmostEqual(metrics['mean_score'], .5)
        self.assertAlmostEqual(metrics['ks_classification'], .5)

    def test_single_class_batch_has_no_auc_but_other_metrics_remain(self):
        metrics = classification_metrics([0, 0, 0], [.1, .6, .4], [0, 1, 0])
        self.assertIsNone(metrics['auc'])
        self.assertIsNone(metrics['ks_classification'])
        self.assertIsNone(metrics['recall'])
        self.assertTrue(np.isfinite(metrics['predicted_positive_rate']))


class RemoteAlertTests(unittest.TestCase):
    """Regras aplicáveis de test_alerts.py; limiares do experimento atual."""
    def test_healthy_batch_raises_no_performance_alert(self):
        self.assertFalse(performance_degraded(.01, .01))
        self.assertFalse(performance_degraded(-.2, -.1))

    def test_auc_drop_alerts_at_threshold_and_not_below(self):
        self.assertFalse(performance_degraded(CONCEPT_AUC_DROP - .001, 0))
        self.assertTrue(performance_degraded(CONCEPT_AUC_DROP + .001, 0))
        self.assertTrue(performance_degraded(.85 - .80, 0))

    def test_recall_drop_alerts_at_threshold_and_not_below(self):
        self.assertFalse(performance_degraded(0, CONCEPT_RECALL_DROP - .001))
        self.assertTrue(performance_degraded(0, CONCEPT_RECALL_DROP + .001))
        self.assertTrue(performance_degraded(0, .77 - .67))

    def test_missing_auc_does_not_break_recall_rule(self):
        self.assertTrue(performance_degraded(None, .2))
        self.assertFalse(performance_degraded(None, .01))
        self.assertFalse(performance_degraded(None, None))
