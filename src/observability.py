"""Coleta pós-fato de eventos e métricas de saúde a partir de artefatos existentes.

Lê manifests/métricas já gravados (só leitura) e escreve exclusivamente em
reports/observability/ (events.jsonl, metrics.json, index.html).
Não regenera drift nem treino.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


DRIFT_MANIFEST = Path("reports/drift/manifest.json")
MODEL_METRICS = Path("models/current/metrics.json")
QUARANTINE_DIR = Path("data/quarantine")
PREDICTIONS_MANIFEST = Path("data/predictions/manifest.json")
SCENARIOS_MANIFEST = Path("data/production/scenarios/manifest.json")
DEFAULT_OUTPUT_DIR = Path("reports/observability")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel_from_abs(path_str: str | None, fallback: str | None = None):
    """Converte path absoluto alienígena em relativo/nome quando possível."""
    if not path_str:
        return fallback
    name = Path(path_str).name
    if fallback:
        return fallback
    return name


def _quarantine_listing(quarantine_dir: Path):
    """Só nomes e contagem — nunca abre CSV/Parquet de quarentena."""
    if not quarantine_dir.is_dir():
        return {"exists": False, "file_count": 0, "files": []}
    files = sorted(p.name for p in quarantine_dir.iterdir() if p.is_file())
    return {"exists": True, "file_count": len(files), "files": files}


def _ingest_status(quarantine: dict):
    if not quarantine["exists"]:
        return "unknown"
    error_files = [f for f in quarantine["files"] if f.endswith("_errors.csv")]
    if error_files:
        return "rejected"
    if quarantine["file_count"] > 0:
        return "ok"
    return "unknown"


def _model_status(metrics_path: Path):
    if not metrics_path.is_file():
        return "unknown", None
    try:
        metrics = _read_json(metrics_path)
    except (OSError, json.JSONDecodeError):
        return "fail", None
    required = ("auc", "recall_at_0.5", "ks")
    if not all(k in metrics for k in required):
        return "fail", metrics
    return "ok", metrics


def _drift_status(manifest_path: Path):
    if not manifest_path.is_file():
        return "unknown", None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return "fail", None
    scenarios = manifest.get("scenarios") or []
    if not scenarios:
        return "fail", manifest
    any_alert = any(bool(s.get("dataset_drift_detected")) for s in scenarios)
    return ("alert" if any_alert else "ok"), manifest


def collect(
    drift_manifest: Path = DRIFT_MANIFEST,
    model_metrics: Path = MODEL_METRICS,
    quarantine_dir: Path = QUARANTINE_DIR,
    predictions_manifest: Path = PREDICTIONS_MANIFEST,
    scenarios_manifest: Path = SCENARIOS_MANIFEST,
    ts: str | None = None,
):
    """Lê fontes e normaliza eventos tipados + metrics.json (contratos §4.1 / §4.2)."""
    ts = ts or _utc_now()
    events = []

    quarantine = _quarantine_listing(Path(quarantine_dir))
    ingest_status = _ingest_status(quarantine)
    events.append({
        "ts": ts,
        "source": "ingest",
        "event": "pipeline_status",
        "status": ingest_status,
        "duration_ms": None,
        "volume": quarantine["file_count"] if quarantine["exists"] else None,
        "refs": {
            "quarantine_dir": str(Path(quarantine_dir).as_posix()),
            "files": quarantine["files"],
        },
    })

    train_status, model = _model_status(Path(model_metrics))
    model_refs = {"metrics_path": str(Path(model_metrics).as_posix())}
    events.append({
        "ts": ts,
        "source": "train",
        "event": "model_snapshot",
        "status": train_status,
        "duration_ms": None,
        "volume": None,
        "refs": model_refs,
    })

    drift_status, drift_manifest_data = _drift_status(Path(drift_manifest))
    drift_scenarios = []
    if drift_manifest_data:
        for scenario in drift_manifest_data.get("scenarios") or []:
            name = scenario.get("name")
            reports = []
            for report in scenario.get("reports") or []:
                reports.append({
                    "file": report.get("file"),
                    "sha256": report.get("sha256"),
                })
            source = scenario.get("source") or {}
            source_name = None
            if isinstance(source, dict) and source.get("path"):
                source_name = _safe_rel_from_abs(
                    source["path"], fallback=f"data/production/scenarios/{name}.parquet" if name else None
                )
            drift_scenarios.append({
                "name": name,
                "rows": scenario.get("rows"),
                "drift_share": scenario.get("drift_share"),
                "dataset_drift_detected": bool(scenario.get("dataset_drift_detected")),
                "drifted_features": list(scenario.get("drifted_features") or []),
            })
            events.append({
                "ts": ts,
                "source": "detect_drift",
                "event": "drift_scenario",
                "status": "alert" if scenario.get("dataset_drift_detected") else "ok",
                "duration_ms": None,
                "volume": scenario.get("rows"),
                "refs": {
                    "name": name,
                    "source_file": source_name,
                    "source_sha256": source.get("sha256") if isinstance(source, dict) else None,
                    "drift_share": scenario.get("drift_share"),
                    "dataset_drift_detected": bool(scenario.get("dataset_drift_detected")),
                    "drifted_features": list(scenario.get("drifted_features") or []),
                    "reports": reports,
                },
            })
        summary = drift_manifest_data.get("summary") or {}
        events.append({
            "ts": ts,
            "source": "detect_drift",
            "event": "pipeline_status",
            "status": drift_status,
            "duration_ms": None,
            "volume": len(drift_scenarios) or None,
            "refs": {
                "manifest_path": str(Path(drift_manifest).as_posix()),
                "created_at_utc": drift_manifest_data.get("created_at_utc"),
                "summary_file": summary.get("file"),
                "summary_sha256": summary.get("sha256"),
                "reference_sha256": (drift_manifest_data.get("reference") or {}).get("sha256"),
                "reference_rows": drift_manifest_data.get("reference_rows"),
            },
        })
    else:
        events.append({
            "ts": ts,
            "source": "detect_drift",
            "event": "pipeline_status",
            "status": drift_status,
            "duration_ms": None,
            "volume": None,
            "refs": {"manifest_path": str(Path(drift_manifest).as_posix())},
        })

    optional_refs = {}
    for label, path in (
        ("predictions_manifest", Path(predictions_manifest)),
        ("scenarios_manifest", Path(scenarios_manifest)),
    ):
        optional_refs[label] = {
            "path": str(path.as_posix()),
            "exists": path.is_file(),
        }
    events.append({
        "ts": ts,
        "source": "observe",
        "event": "pipeline_status",
        "status": "ok",
        "duration_ms": None,
        "volume": None,
        "refs": optional_refs,
    })

    dataset_drift_count = sum(1 for s in drift_scenarios if s.get("dataset_drift_detected"))
    metrics = {
        "generated_at_utc": ts,
        "pipeline": {
            "stages": [
                {
                    "name": "ingest",
                    "status": ingest_status,
                    "volume": quarantine["file_count"] if quarantine["exists"] else None,
                    "duration_ms": None,
                },
                {
                    "name": "train",
                    "status": train_status,
                    "volume": None,
                    "duration_ms": None,
                },
                {
                    "name": "detect_drift",
                    "status": drift_status,
                    "volume": len(drift_scenarios) or None,
                    "duration_ms": None,
                },
            ]
        },
        "model": {
            "metrics_path": str(Path(model_metrics).as_posix()),
            "auc": (model or {}).get("auc"),
            "recall_at_0.5": (model or {}).get("recall_at_0.5"),
            "ks": (model or {}).get("ks"),
        },
        "drift": {
            "manifest_path": str(Path(drift_manifest).as_posix()),
            "created_at_utc": (drift_manifest_data or {}).get("created_at_utc"),
            "scenarios": drift_scenarios,
            "alerts": {
                "dataset_drift_count": dataset_drift_count,
                "any_alert": dataset_drift_count > 0,
            },
        },
        "contract_alerts": {
            "quarantine_file_count": quarantine["file_count"] if quarantine["exists"] else 0,
        },
    }
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


def _evidently_href(scenario_name: str | None) -> str | None:
    """Link relativo de reports/observability/ → reports/drift/<name>.html."""
    if not scenario_name:
        return None
    return f"../drift/{scenario_name}.html"


def render_html(metrics: dict, events: list | None = None) -> str:
    """Gera index.html a partir de metrics (+ amostra de events). Sem deps externas."""
    events = events or []
    generated = _esc(metrics.get("generated_at_utc"))
    stages = (metrics.get("pipeline") or {}).get("stages") or []
    model = metrics.get("model") or {}
    drift = metrics.get("drift") or {}
    alerts = drift.get("alerts") or {}
    contract = metrics.get("contract_alerts") or {}
    scenarios = drift.get("scenarios") or []

    stage_rows = []
    for stage in stages:
        status = stage.get("status") or "unknown"
        stage_rows.append(
            "<tr>"
            f"<td>{_esc(stage.get('name'))}</td>"
            f"<td><span class=\"badge {_status_class(status)}\">{_esc(status)}</span></td>"
            f"<td>{_esc(stage.get('volume'))}</td>"
            f"<td>{_esc(stage.get('duration_ms'))}</td>"
            "</tr>"
        )
    if not stage_rows:
        stage_rows.append('<tr><td colspan="4">Nenhuma etapa coletada</td></tr>')

    scenario_rows = []
    for scenario in scenarios:
        name = scenario.get("name")
        href = _evidently_href(name)
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

    sample = events[:12]
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
            "</tr>"
        )
    if not event_rows:
        event_rows.append('<tr><td colspan="5">Nenhum evento</td></tr>')

    any_alert = bool(alerts.get("any_alert"))
    drift_count = alerts.get("dataset_drift_count")
    quarantine_count = contract.get("quarantine_file_count")
    alert_banner_class = "alert" if any_alert or (quarantine_count or 0) > 0 else "ok"

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
    main {{ max-width: 960px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }}
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
    table {{
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
    <p class="meta">Snapshot gerado em {generated} (UTC) · pasta <code>reports/observability/</code></p>

    <div class="banner {alert_banner_class}">
      <strong>Alertas:</strong>
      dataset_drift = {_esc(drift_count)} cenário(s);
      any_alert = {_esc(any_alert)};
      quarentena (contrato) = {_esc(quarantine_count)} arquivo(s).
    </div>

    <div class="grid">
      <div class="stat"><span class="label">AUC</span><span class="value">{_esc(model.get('auc'))}</span></div>
      <div class="stat"><span class="label">Recall @ 0.5</span><span class="value">{_esc(model.get('recall_at_0.5'))}</span></div>
      <div class="stat"><span class="label">KS</span><span class="value">{_esc(model.get('ks'))}</span></div>
      <div class="stat"><span class="label">Cenários drift</span><span class="value">{_esc(len(scenarios))}</span></div>
    </div>

    <h2>Status das etapas</h2>
    <table>
      <thead><tr><th>Etapa</th><th>Status</th><th>Volume</th><th>Duração (ms)</th></tr></thead>
      <tbody>
        {''.join(stage_rows)}
      </tbody>
    </table>

    <h2>Métricas do modelo</h2>
    <table>
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
    <table>
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
    <table>
      <thead><tr><th>Indicador</th><th>Valor</th></tr></thead>
      <tbody>
        <tr><td>quarantine_file_count</td><td>{_esc(quarantine_count)}</td></tr>
        <tr><td>drift.manifest_path</td><td><code>{_esc(drift.get('manifest_path'))}</code></td></tr>
        <tr><td>drift.created_at_utc</td><td>{_esc(drift.get('created_at_utc'))}</td></tr>
      </tbody>
    </table>

    <h2>Amostra de eventos</h2>
    <table>
      <thead><tr><th>ts</th><th>source</th><th>event</th><th>status</th><th>volume</th></tr></thead>
      <tbody>
        {''.join(event_rows)}
      </tbody>
    </table>

    <footer>
      Gerado por <code>python -m src.observability</code> / <code>make observe</code>.
      Escrita exclusiva sob <code>reports/observability/</code>.
    </footer>
  </main>
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
    lines = [
        json.dumps(event, ensure_ascii=False, allow_nan=False)
        for event in payload["events"]
    ]
    events_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(payload["metrics"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    index_path.write_text(
        render_html(payload["metrics"], payload["events"]),
        encoding="utf-8",
    )
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--drift-manifest", default=str(DRIFT_MANIFEST))
    parser.add_argument("--model-metrics", default=str(MODEL_METRICS))
    parser.add_argument("--quarantine-dir", default=str(QUARANTINE_DIR))
    parser.add_argument("--predictions-manifest", default=str(PREDICTIONS_MANIFEST))
    parser.add_argument("--scenarios-manifest", default=str(SCENARIOS_MANIFEST))
    args = parser.parse_args(argv)
    payload, paths = run(
        output_dir=Path(args.output_dir),
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
