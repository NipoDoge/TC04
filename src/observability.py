"""Consolida logs reais de execução, infraestrutura e resultados offline de drift.

Lê manifests/métricas já gravados (só leitura) e escreve exclusivamente em
reports/observability/ (events.jsonl, metrics.json, index.html).
Não regenera drift nem treino.
"""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.execution_log import DEFAULT_LOG, infrastructure


DRIFT_MANIFEST = Path("reports/drift/manifest.json")
MODEL_METRICS = Path("models/current/metrics.json")
QUARANTINE_DIR = Path("data/quarantine")
PREDICTIONS_MANIFEST = Path("data/predictions/manifest.json")
SCENARIOS_MANIFEST = Path("data/production/scenarios/manifest.json")
DEFAULT_OUTPUT_DIR = Path("reports/observability")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value):
    raise ValueError(f"Constante JSON inválida: {value}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _number(value, low=0, high=math.inf):
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and low <= value <= high)
    except OverflowError:
        return False


def _manifest_status(path, collection):
    """Existência de arquivo não comprova que um manifesto possa ser consumido."""
    path = Path(path)
    if not path.exists():
        return "unknown"
    try:
        data = _read_json(path)
        entries = data.get(collection) if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries or not all(
            isinstance(entry, dict) and isinstance(entry.get("name"), str)
            and _number(entry.get("rows")) and isinstance(entry["rows"], int)
            for entry in entries
        ):
            return "fail"
    except (OSError, ValueError, UnicodeError):
        return "fail"
    return "ok"


def _model_status(metrics_path: Path):
    if not metrics_path.is_file():
        return "unknown", None
    try:
        metrics = _read_json(metrics_path)
    except (OSError, ValueError, UnicodeError):
        return "fail", None
    required = ("auc", "recall_at_0.5", "ks")
    if not isinstance(metrics, dict) or not all(
        _number(metrics.get(k), high=1)
        for k in required
    ):
        return "fail", None
    return "ok", metrics


def _drift_status(manifest_path: Path):
    if not manifest_path.is_file():
        return "unknown", None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, UnicodeError):
        return "fail", None
    if not isinstance(manifest, dict):
        return "fail", None
    scenarios = manifest.get("scenarios") or []
    if not isinstance(scenarios, list) or not scenarios or not all(
        isinstance(s, dict) and isinstance(s.get("dataset_drift_detected"), bool)
        and isinstance(s.get("name"), str) and isinstance(s.get("rows"), int)
        and _number(s["rows"]) and _number(s.get("drift_share"), high=1)
        and isinstance(s.get("drifted_features"), list)
        and all(isinstance(feature, str) for feature in s["drifted_features"])
        for s in scenarios
    ):
        return "fail", None
    any_alert = any(bool(s.get("dataset_drift_detected")) for s in scenarios)
    return ("alert" if any_alert else "ok"), manifest


STAGES = ("prepare", "ingest", "train", "simulate_drift", "predict", "detect_drift", "concept_drift")


def read_executions(path):
    """Leitura consistente do journal; linhas inválidas aparecem como falha de coleta."""
    events, invalid = [], []
    path = Path(path)
    if path.is_file():
        with path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line, parse_constant=_reject_constant)
                    if (not isinstance(event, dict)
                            or event.get("event") not in {"execution_started", "execution_finished"}
                            or event.get("source") not in STAGES
                            or not all(isinstance(event.get(k), str) and event[k] for k in ("execution_id", "run_id", "ts", "started_at_utc"))):
                        raise ValueError("Evento inválido")
                    for key in ("ts", "started_at_utc"):
                        if datetime.fromisoformat(event[key]).tzinfo is None:
                            raise ValueError("Horário sem fuso")
                    if datetime.fromisoformat(event["ts"]) < datetime.fromisoformat(event["started_at_utc"]):
                        raise ValueError("Evento anterior ao início")
                    if "infrastructure" in event and not isinstance(event["infrastructure"], dict):
                        raise ValueError("Recursos inválidos")
                    if event.get("volume") is not None and not (
                        isinstance(event["volume"], int) and _number(event["volume"])
                    ):
                        raise ValueError("Volume inválido")
                    if event["event"] == "execution_started" and event.get("status") != "running":
                        raise ValueError("Início inválido")
                    if event["event"] == "execution_finished":
                        if (event.get("status") not in {"ok", "rejected", "fail", "alert"}
                                or not _number(event.get("duration_ms"))):
                            raise ValueError("Término inválido")
                    events.append(event)
                except (ValueError, TypeError):
                    invalid.append(number)
    return events, invalid


def collect(
    drift_manifest: Path = DRIFT_MANIFEST,
    model_metrics: Path = MODEL_METRICS,
    quarantine_dir: Path = QUARANTINE_DIR,
    predictions_manifest: Path = PREDICTIONS_MANIFEST,
    scenarios_manifest: Path = SCENARIOS_MANIFEST,
    execution_log=None,
    concept_manifest: Path = Path("artifacts/concept_drift/manifest.json"),
    run_id: str | None = None,
    stale_after_seconds: float = 86400,
    ts: str | None = None,
):
    """Separa execução real, saúde do coletor e resultados offline dos experimentos."""
    if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0:
        raise ValueError("O prazo de validade deve ser positivo e finito")
    ts = ts or _utc_now()
    clock = datetime.fromisoformat(ts)
    if clock.tzinfo is None:
        raise ValueError("O horário da coleta precisa incluir o fuso")
    execution_log = Path(execution_log or os.environ.get("TC04_EXECUTION_LOG", DEFAULT_LOG))
    events, invalid = read_executions(execution_log)
    if run_id:
        events = [e for e in events if e["run_id"] == run_id]
    executions = {}
    for event in events:
        executions[event["execution_id"]] = event
    latest = {}
    for event in sorted(executions.values(), key=lambda e: datetime.fromisoformat(e["started_at_utc"])):
        latest[event["source"]] = event
    stages = []
    for name in STAGES:
        event = latest.get(name, {})
        status = event.get("status", "unknown")
        age = (clock - datetime.fromisoformat(event["ts"])).total_seconds() if event else None
        if age is not None and age > (300 if status == "running" else stale_after_seconds):
            status = "stale"
        stages.append({
            "name": name, "status": status, "volume": event.get("volume"),
            "volume_unit": "rows", "duration_ms": event.get("duration_ms"),
            "run_id": event.get("run_id"), "execution_id": event.get("execution_id"),
            "ts": event.get("ts"), "error_type": event.get("error_type"),
            "cpu_seconds": event.get("cpu_seconds"),
            "process_peak_rss_mb": (event.get("infrastructure") or {}).get("process_peak_rss_mb"),
        })
    counts = {key: sum(e.get("status") == key for e in executions.values())
              for key in ("ok", "fail", "rejected", "alert", "running")}
    model_status, model = _model_status(Path(model_metrics))
    drift_status, drift = _drift_status(Path(drift_manifest))
    drift_scenarios = [{k: s.get(k) for k in ("name", "rows", "drift_share", "dataset_drift_detected", "drifted_features")}
                       for s in (drift or {}).get("scenarios", [])]
    concept_path = Path(concept_manifest)
    performance, performance_status = [], "unknown"
    if concept_path.is_file():
        try:
            concept = _read_json(concept_path)
            performance = concept["results"]
            if not isinstance(performance, list) or not performance or not all(
                isinstance(r, dict) and isinstance(r.get("performance_alert"), bool)
                and isinstance(r.get("scenario"), str)
                and all(r.get(k) is None or _number(r[k], high=1) for k in ("auc", "recall"))
                and all(r.get(k) is None or _number(r[k], low=-1, high=1) for k in ("auc_drop", "recall_drop"))
                for r in performance
            ):
                raise ValueError("Resultados ausentes")
            performance_status = "alert" if any(r["performance_alert"] for r in performance) else "ok"
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            performance, performance_status = [], "fail"
    artifact_statuses = {
        "predictions_manifest_status": _manifest_status(predictions_manifest, "datasets"),
        "scenarios_manifest_status": _manifest_status(scenarios_manifest, "periods"),
    }
    host = infrastructure()
    host["disk_free_share"] = host["disk_free_bytes"] / host["disk_total_bytes"]
    host["status"] = "alert" if host["disk_free_share"] < 0.05 else "ok"
    statuses = [s["status"] for s in stages]
    if invalid or "fail" in statuses or "rejected" in statuses or "stale" in statuses or "fail" in (model_status, drift_status, performance_status, *artifact_statuses.values()):
        health = "fail"
    elif "unknown" in statuses or "running" in statuses or "unknown" in (model_status, drift_status, performance_status):
        health = "incomplete"
    elif "alert" in statuses or "alert" in (drift_status, performance_status, host["status"]):
        health = "alert"
    else:
        health = "ok"
    quarantine = Path(quarantine_dir)
    metrics = {
        "generated_at_utc": ts,
        "pipeline": {"status": health, "run_id_filter": run_id, "stages": stages,
                     "execution_counts": counts, "execution_count": len(executions),
                     "stale_after_seconds": stale_after_seconds},
        "collector": {"status": "fail" if invalid else "ok", "execution_log": str(execution_log),
                      "invalid_lines": invalid},
        "infrastructure": host,
        "model": {"status": model_status, "metrics_path": str(model_metrics),
                  **{k: (model or {}).get(k) for k in ("auc", "recall_at_0.5", "ks")}},
        "drift": {"status": drift_status, "manifest_path": str(drift_manifest),
                  "created_at_utc": (drift or {}).get("created_at_utc"), "scenarios": drift_scenarios,
                  "alerts": {"dataset_drift_count": sum(bool(s["dataset_drift_detected"]) for s in drift_scenarios),
                             "any_alert": any(s["dataset_drift_detected"] for s in drift_scenarios)}},
        "performance": {"status": performance_status, "manifest_path": str(concept_path),
                        "scenarios": performance, "alert_count": sum(bool(r["performance_alert"]) for r in performance)},
        "contract_alerts": {"quarantined_batch_count": sum(1 for _ in quarantine.glob("*.parquet")) if quarantine.is_dir() else 0,
                            "rejected_execution_count": counts["rejected"]},
        "artifacts": {**artifact_statuses,
                      "predictions_manifest_exists": Path(predictions_manifest).is_file(),
                      "scenarios_manifest_exists": Path(scenarios_manifest).is_file()},
    }
    if "unknown" in artifact_statuses.values() and health in {"ok", "alert"}:
        metrics["pipeline"]["status"] = "incomplete"
    return {"events": events, "metrics": metrics}


def _fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _esc(value):
    return html.escape(_fmt(value), quote=True)


def _status_class(status: str | None) -> str:
    key = (status or "unknown").lower()
    mapping = {
        "ok": "ok",
        "fail": "fail",
        "rejected": "rejected",
        "alert": "alert",
        "unknown": "unknown",
    }
    return mapping.get(key, "unknown")


def render_html(metrics: dict, events: list | None = None, output_dir=DEFAULT_OUTPUT_DIR) -> str:
    """Gera index.html a partir de metrics (+ amostra de events). Sem deps externas."""
    events = events or []
    generated = _esc(metrics.get("generated_at_utc"))
    stages = (metrics.get("pipeline") or {}).get("stages") or []
    model = metrics.get("model") or {}
    drift = metrics.get("drift") or {}
    alerts = drift.get("alerts") or {}
    contract = metrics.get("contract_alerts") or {}
    scenarios = drift.get("scenarios") or []
    sources = metrics.get("artifacts") or {}
    source_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(status)}</td></tr>"
        for label, status in (
            ("Journal", (metrics.get("collector") or {}).get("status")),
            ("Métricas do modelo", model.get("status")),
            ("Manifesto de data drift", drift.get("status")),
            ("Manifesto de concept drift", (metrics.get("performance") or {}).get("status")),
            ("Manifesto de predições", sources.get("predictions_manifest_status")),
            ("Manifesto de cenários", sources.get("scenarios_manifest_status")),
        )
    )

    pipeline = metrics.get("pipeline") or {}
    infra = metrics.get("infrastructure") or {}
    performance = metrics.get("performance") or {}
    perf_rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(r.get(k))}</td>" for k in
                         ("scenario", "auc", "recall", "auc_drop", "recall_drop", "performance_alert")) + "</tr>"
        for r in performance.get("scenarios", [])
    )
    def report_href(path):
        return html.escape(os.path.relpath(path, output_dir), quote=True)
    concept_href = report_href(Path(performance.get("manifest_path", "artifacts/concept_drift/manifest.json")).parent / "analysis.html")
    stage_rows = []
    for stage in stages:
        status = stage.get("status") or "unknown"
        stage_rows.append(
            "<tr>"
            f"<td>{_esc(stage.get('name'))}</td>"
            f"<td><span class=\"badge {_status_class(status)}\">{_esc(status)}</span></td>"
            f"<td>{_esc(stage.get('volume'))}</td>"
            f"<td>{_esc(stage.get('duration_ms'))}</td>"
            f"<td>{_esc(stage.get('cpu_seconds'))}</td>"
            f"<td>{_esc(stage.get('process_peak_rss_mb'))}</td>"
            f"<td>{_esc(stage.get('ts'))}</td>"
            f"<td>{_esc(stage.get('error_type'))}</td>"
            "</tr>"
        )
    if not stage_rows:
        stage_rows.append('<tr><td colspan="4">Nenhuma etapa coletada</td></tr>')

    scenario_rows = []
    for scenario in scenarios:
        name = scenario.get("name")
        href = os.path.relpath(Path(drift.get("manifest_path", DRIFT_MANIFEST)).parent / f"{name}.html", output_dir) if name else None
        link = (
            f'<a href="{html.escape(href, quote=True)}">{_esc(name)}.html</a>'
            if href
            else "—"
        )
        flagged = bool(scenario.get("dataset_drift_detected"))
        features = ", ".join(scenario.get("drifted_features") or []) or "—"
        scenario_rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td>{_esc(scenario.get('rows'))}</td>"
            f"<td>{_esc(scenario.get('drift_share'))}</td>"
            f"<td><span class=\"badge {'alert' if flagged else 'ok'}\">"
            f"{'sim' if flagged else 'não'}</span></td>"
            f"<td>{_esc(features)}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    if not scenario_rows:
        scenario_rows.append('<tr><td colspan="6">Nenhum cenário de drift</td></tr>')

    sample = list(reversed(events))
    event_rows = []
    for event in sample:
        event_rows.append(
            "<tr>"
            f"<td>{_esc(event.get('ts'))}</td>"
            f"<td>{_esc(event.get('source'))}</td>"
            f"<td>{_esc(event.get('event'))}</td>"
            f"<td><span class=\"badge {_status_class(event.get('status'))}\">"
            f"{_esc(event.get('status'))}</span></td>"
            f"<td>{_esc(event.get('volume'))}</td>"
            f"<td>{_esc(event.get('duration_ms'))}</td>"
            f"<td>{_esc(event.get('run_id'))}</td>"
            f"<td>{_esc(event.get('execution_id'))}</td>"
            "</tr>"
        )
    if not event_rows:
        event_rows.append('<tr><td colspan="5">Nenhum evento</td></tr>')

    drift_count = alerts.get("dataset_drift_count")
    quarantine_count = contract.get("quarantined_batch_count")
    alert_banner_class = "alert" if pipeline.get("status") != "ok" else "ok"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Observabilidade — saúde do pipeline</title>
  <style>
    :root {{
      --bg: #f7f4ef;
      --ink: #1c1917;
      --muted: #57534e;
      --line: #d6d3d1;
      --ok: #166534;
      --ok-bg: #dcfce7;
      --fail: #991b1b;
      --fail-bg: #fee2e2;
      --alert: #9a3412;
      --alert-bg: #ffedd5;
      --rejected: #7c2d12;
      --rejected-bg: #fed7aa;
      --unknown: #44403c;
      --unknown-bg: #e7e5e4;
      --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #ebe6de 0%, var(--bg) 180px);
      line-height: 1.45;
    }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.6rem; }}
    h2 {{ margin: 1.75rem 0 0.6rem; font-size: 1.15rem; border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; }}
    p.meta {{ margin: 0; color: var(--muted); font-size: 0.92rem; }}
    .banner {{
      margin-top: 1rem;
      padding: 0.85rem 1rem;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--card);
    }}
    .banner.alert {{ border-color: #fdba74; background: var(--alert-bg); }}
    .banner.ok {{ border-color: #86efac; background: var(--ok-bg); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin-top: 0.9rem;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.75rem 0.85rem;
    }}
    .stat .label {{ display: block; color: var(--muted); font-size: 0.8rem; }}
    .stat .value {{ display: block; font-size: 1.25rem; font-weight: 650; margin-top: 0.15rem; }}
    table {{ display: block; overflow-x: auto;
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.45rem 0.55rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ background: #f5f5f4; font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    .badge {{
      display: inline-block;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .badge.ok {{ color: var(--ok); background: var(--ok-bg); }}
    .badge.fail {{ color: var(--fail); background: var(--fail-bg); }}
    .badge.alert {{ color: var(--alert); background: var(--alert-bg); }}
    .badge.rejected {{ color: var(--rejected); background: var(--rejected-bg); }}
    .badge.unknown {{ color: var(--unknown); background: var(--unknown-bg); }}
    a {{ color: #1d4ed8; }}
    footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <main>
    <h1>Observabilidade do pipeline</h1>
    <p><strong>Saúde: {_esc(pipeline.get('status'))}</strong> · Execuções: {_esc(pipeline.get('execution_count'))}
    · Histórico de resultados: {_esc(pipeline.get('execution_counts'))}</p>
    <p>Última execução de cada etapa; falhas anteriores permanecem no histórico.
    Filtro de execução: {_esc(pipeline.get('run_id_filter') or 'todas')}.
    Datas em UTC. Etapas sem registro ficam unknown; registros vencidos ficam stale.</p>
    <p class="meta">Snapshot gerado em {generated} (UTC) · pasta <code>reports/observability/</code></p>

    <div class="banner {alert_banner_class}">
      <strong>Alertas:</strong>
      drift de dados = {_esc(drift_count)} cenário(s);
      estado do coletor = {_esc(metrics.get('collector', {}).get('status'))};
      quarentena (contrato) = {_esc(quarantine_count)} lote(s);
      alertas de desempenho = {_esc(performance.get('alert_count'))}.
    </div>

    <div class="grid">
      <div class="stat"><span class="label">AUC</span><span class="value">{_esc(model.get('auc'))}</span></div>
      <div class="stat"><span class="label">Recall @ 0.5</span><span class="value">{_esc(model.get('recall_at_0.5'))}</span></div>
      <div class="stat"><span class="label">KS</span><span class="value">{_esc(model.get('ks'))}</span></div>
      <div class="stat"><span class="label">Cenários drift</span><span class="value">{_esc(len(scenarios))}</span></div>
    </div>

    <h2>Últimas execuções reais</h2>
    <table aria-label="Últimas execuções reais">
      <thead><tr><th>Etapa</th><th>Status</th><th>Linhas</th><th>Duração (ms)</th><th>CPU (s)</th><th>Pico RSS (MB)</th><th>Horário UTC</th><th>Tipo de erro</th></tr></thead>
      <tbody>
        {''.join(stage_rows)}
      </tbody>
    </table>

    <h2>Infraestrutura no momento da coleta</h2>
    <p>Disco livre: {_esc(infra.get('disk_free_bytes'))} bytes de {_esc(infra.get('disk_total_bytes'))};
    fração livre: {_esc(infra.get('disk_free_share'))}; CPUs: {_esc(infra.get('cpu_count'))}; estado: {_esc(infra.get('status'))}.
    Alerta de disco abaixo de 5% livre. CPU é tempo consumido pela etapa; RSS é o pico do processo até aquele momento.</p>
    <h2>Desempenho sob concept drift</h2>
    <p>Estado: {_esc(performance.get('status'))}. Experimento sintético offline com rótulos alterados;
    compara cada cenário ao controle do mesmo holdout. <a href="{concept_href}">Análise completa</a>.</p>
    <table aria-label="Desempenho sob concept drift"><thead><tr><th>Cenário</th><th>AUC</th><th>Recall</th><th>Queda AUC</th><th>Queda recall</th><th>Alerta</th></tr></thead>
    <tbody>{perf_rows}</tbody></table>
    <h2>Métricas publicadas do baseline (validação offline)</h2>
    <table aria-label="Métricas publicadas do baseline">
      <thead><tr><th>Campo</th><th>Valor</th></tr></thead>
      <tbody>
        <tr><td>metrics_path</td><td><code>{_esc(model.get('metrics_path'))}</code></td></tr>
        <tr><td>auc</td><td>{_esc(model.get('auc'))}</td></tr>
        <tr><td>recall_at_0.5</td><td>{_esc(model.get('recall_at_0.5'))}</td></tr>
        <tr><td>ks</td><td>{_esc(model.get('ks'))}</td></tr>
      </tbody>
    </table>

    <h2>Drift por cenário</h2>
    <p class="meta">Links relativos para HTML Evidently existentes (somente leitura; pasta <code>reports/drift/</code> não é regenerada).</p>
    <table aria-label="Drift por cenário">
      <thead>
        <tr>
          <th>Cenário</th><th>Linhas</th><th>drift_share</th>
          <th>dataset_drift</th><th>features</th><th>Evidently</th>
        </tr>
      </thead>
      <tbody>
        {''.join(scenario_rows)}
      </tbody>
    </table>

    <h2>Alertas de contrato / quarentena</h2>
    <table aria-label="Alertas de contrato e quarentena">
      <thead><tr><th>Indicador</th><th>Valor</th></tr></thead>
      <tbody>
        <tr><td>quarantined_batch_count</td><td>{_esc(quarantine_count)}</td></tr>
        <tr><td>drift.manifest_path</td><td><code>{_esc(drift.get('manifest_path'))}</code></td></tr>
        <tr><td>drift.created_at_utc</td><td>{_esc(drift.get('created_at_utc'))}</td></tr>
      </tbody>
    </table>

    <h2>Integridade das fontes de monitoramento</h2>
    <table aria-label="Integridade das fontes"><thead><tr><th>Fonte</th><th>Status</th></tr></thead>
    <tbody>{source_rows}</tbody></table>
    <h2>Histórico de eventos reais</h2>
    <p>Coletor: {_esc(metrics.get('collector', {}).get('status'))}; linhas inválidas: {_esc(metrics.get('collector', {}).get('invalid_lines'))}.</p>
    <label>Filtrar eventos <input id="event-filter" placeholder="etapa, status ou identificador" oninput="filterEvents(this.value)"></label>
    <table aria-label="Histórico de eventos reais">
      <thead><tr><th>Horário UTC</th><th>Etapa</th><th>Evento</th><th>Status</th><th>Linhas</th><th>Duração (ms)</th><th>Run ID</th><th>Execution ID</th></tr></thead>
      <tbody id="event-history">
        {''.join(event_rows)}
      </tbody>
    </table>

    <footer>
      Gerado por <code>python -m src.observability</code> / <code>make observe</code>.
      O journal de execução é preservado; este painel é atualizado sob demanda.
    </footer>
  </main>
<script>
function filterEvents(value) {{
  for (const row of document.querySelectorAll('#event-history tr')) {{
    row.hidden = !row.textContent.toLowerCase().includes(value.toLowerCase());
  }}
}}
</script>
</body>
</html>
"""


def write_artifacts(payload, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Persiste events.jsonl, metrics.json e index.html sob reports/observability/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    metrics_path = output_dir / "metrics.json"
    index_path = output_dir / "index.html"
    journal = Path(payload["metrics"]["collector"]["execution_log"]).resolve()
    if journal in {p.resolve() for p in (events_path, metrics_path, index_path)}:
        raise ValueError("O journal deve ser separado das saídas regeneráveis do painel")
    lines = [
        json.dumps(event, ensure_ascii=False, allow_nan=False)
        for event in payload["events"]
    ]
    # Renderizar e serializar antes de substituir qualquer arquivo publicado.
    contents = {
        events_path: "\n".join(lines) + ("\n" if lines else ""),
        metrics_path: json.dumps(payload["metrics"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        index_path: render_html(payload["metrics"], payload["events"], output_dir),
    }
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=output_dir) as temporary:
        for destination, content in contents.items():
            (Path(temporary) / destination.name).write_text(content, encoding="utf-8")
        for destination in contents:
            os.replace(Path(temporary) / destination.name, destination)
    return {
        "events_path": events_path,
        "metrics_path": metrics_path,
        "index_path": index_path,
    }


def run(output_dir: Path = DEFAULT_OUTPUT_DIR, **collect_kwargs):
    """Coleta + escrita dos três artefatos (events, metrics, index.html)."""
    payload = collect(**collect_kwargs)
    paths = write_artifacts(payload, output_dir=output_dir)
    return payload, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-log")
    parser.add_argument("--concept-manifest", default="artifacts/concept_drift/manifest.json")
    parser.add_argument("--run-id")
    parser.add_argument("--stale-after-seconds", type=float, default=86400)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--drift-manifest", default=str(DRIFT_MANIFEST))
    parser.add_argument("--model-metrics", default=str(MODEL_METRICS))
    parser.add_argument("--quarantine-dir", default=str(QUARANTINE_DIR))
    parser.add_argument("--predictions-manifest", default=str(PREDICTIONS_MANIFEST))
    parser.add_argument("--scenarios-manifest", default=str(SCENARIOS_MANIFEST))
    args = parser.parse_args(argv)
    payload, paths = run(
        output_dir=Path(args.output_dir),
        execution_log=args.execution_log, concept_manifest=Path(args.concept_manifest),
        run_id=args.run_id, stale_after_seconds=args.stale_after_seconds,
        drift_manifest=Path(args.drift_manifest),
        model_metrics=Path(args.model_metrics),
        quarantine_dir=Path(args.quarantine_dir),
        predictions_manifest=Path(args.predictions_manifest),
        scenarios_manifest=Path(args.scenarios_manifest),
    )
    stages = payload["metrics"]["pipeline"]["stages"]
    print(f"eventos={len(payload['events'])} → {paths['events_path']}")
    print(f"métricas → {paths['metrics_path']}")
    print(f"dashboard → {paths['index_path']}")
    for stage in stages:
        print(f"stage {stage['name']}: status={stage['status']} volume={stage['volume']}")
    alerts = payload["metrics"]["drift"]["alerts"]
    print(
        f"drift alerts: any={alerts['any_alert']} "
        f"dataset_drift_count={alerts['dataset_drift_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
