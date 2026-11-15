"""Pipeline de monitoramento: processa os lotes de produção na ordem de chegada.

Para cada lote: contrato de dados (ingestão) -> escoragem -> drift (Evidently)
-> desempenho com rótulos maturados -> regras de alerta. Cada etapa grava um
evento no log central; métricas e relatórios vão para o MLflow; o consolidado
fica em ``reports/monitoring_summary.json`` e alimenta o dashboard.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from statistics import median
import sys

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from src import alerts as alert_rules
from src import config, tracking
from src.batch_drift import drift_report, load_model, score
from src.ingest import OK, STATUS, validate_batch
from src.observability import log_event, new_run_id, setup_logging, track

log = logging.getLogger("monitor")
EVENT_LEVEL = {"critical": "CRITICAL", "alert": "ALERT", "warning": "WARNING"}


def performance(scored):
    """Métricas do modelo em um lote já escorado e com rótulo observado."""
    y, p = scored[config.TARGET], scored[config.SCORE]
    flagged = p >= config.THRESHOLD
    result = {
        "auc": None, "ks": None,
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "default_rate": float(y.mean()),
        "mean_score": float(p.mean()),
        "flagged_rate": float(flagged.mean()),
    }
    if y.nunique() == 2:
        fpr, tpr, _ = roc_curve(y, p)
        result.update(auc=float(roc_auc_score(y, p)), ks=float((tpr - fpr).max()))
    return result


def load_baseline(reference_scored, metrics_path=config.BASELINE_METRICS):
    """AUC, recall e KS vêm da validação (fora do treino); as taxas, da referência."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    observed = performance(reference_scored)
    return {"auc": metrics["auc"], "recall": metrics[config.RECALL_KEY], "ks": metrics["ks"],
            "default_rate": observed["default_rate"], "mean_score": observed["mean_score"],
            "flagged_rate": observed["flagged_rate"], "rows": len(reference_scored)}


def run_stage(result, name, action):
    """Executa uma etapa medindo duração e registrando o desfecho no log central."""
    try:
        with track(name, batch=result["batch"]) as info:
            value = action(info)
    except Exception:
        result["stages"][name] = {"status": "failed", "duration_s": None}
        raise
    result["stages"][name] = {"status": "success", "duration_s": info["duration_s"]}
    return value


def process_batch(entry, *, model, reference, baseline, quarantine_dir, previous_durations=()):
    """Processa um lote e devolve o registro consolidado (sem dados de clientes)."""
    batch, path = entry["batch"], Path(entry["file"])
    columns = list(model.feature_names_in_) + [config.SCORE]
    result = {"batch": batch, "month": entry.get("month"), "intensity": entry.get("intensity"),
              "kind": entry.get("kind", "regular"), "stages": {}, "alerts": []}

    ingest = validate_batch(path, quarantine_dir)
    result["stages"]["ingest"] = {"status": STATUS[ingest.code], "duration_s": ingest.duration_s}
    result.update(status=STATUS[ingest.code], rows=ingest.rows, duration_s=ingest.duration_s)
    if ingest.code != OK:
        result.update(violations=ingest.violations, quarantine=ingest.quarantine, error=ingest.error)
        result["alerts"] = alert_rules.evaluate(
            batch, ingest_code=ingest.code, violations=ingest.violations, error=ingest.error)
        return result

    try:
        def do_score(info):
            scored = score(model, pd.read_parquet(path))
            config.SCORED_DIR.mkdir(parents=True, exist_ok=True)
            scored.to_parquet(config.SCORED_DIR / f"{batch}.parquet")
            info["rows"] = len(scored)
            return scored

        def do_drift(info):
            html = config.BATCH_REPORTS / f"{batch}.html"
            summary = drift_report(reference, scored, columns, html)
            info.update(rows=len(scored), report=str(html),
                        max_psi=max(item["psi"] for item in summary.values()),
                        alert=[c for c, item in summary.items() if item["status"] == "alert"],
                        warning=[c for c, item in summary.items() if item["status"] == "warning"])
            result["report"] = str(html)
            return summary

        def do_evaluate(info):
            metrics = performance(scored)
            info.update(rows=len(scored), **metrics)
            return metrics

        scored = run_stage(result, "score", do_score)
        result["drift"] = run_stage(result, "drift", do_drift)
        result["performance"] = run_stage(result, "evaluate", do_evaluate)
    except Exception as error:
        log.error("batch=%s falha operacional: %s", batch, error)
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["alerts"] = alert_rules.evaluate(batch, ingest_code=2, error=result["error"])
        return result

    result["duration_s"] = sum(stage["duration_s"] for stage in result["stages"].values())
    result["alerts"] = alert_rules.evaluate(
        batch, rows=result["rows"], drift=result["drift"], performance=result["performance"],
        baseline=baseline, duration_s=result["duration_s"], previous_durations=previous_durations)
    return result


def batch_metrics(result):
    """Métricas numéricas de um lote, no formato registrado no MLflow."""
    counts = alert_rules.count_by_severity(result["alerts"])
    metrics = {"linhas": result["rows"] or 0, "duracao_s": result["duration_s"] or 0.0,
               "lote_rejeitado": float(result["status"] != "success"),
               **{f"alertas_{severity}": count for severity, count in counts.items()}}
    if result["status"] == "success":
        metrics.update({name: value for name, value in result["performance"].items() if value is not None})
        metrics.update({f"psi.{column}": item["psi"] for column, item in result["drift"].items()})
        metrics["features_em_alerta"] = sum(
            item["status"] == "alert" for column, item in result["drift"].items() if column != config.SCORE)
    return metrics


def track_batch(mlflow, result, step):
    """Um run aninhado por lote (com o relatório) e a série temporal no run pai."""
    metrics = batch_metrics(result)
    worst = next((s for s in alert_rules.SEVERITIES if any(a["severity"] == s for a in result["alerts"])), "ok")
    with mlflow.start_run(run_name=result["batch"], nested=True):
        mlflow.set_tags({"status": result["status"], "mes": result["month"], "tipo": result["kind"],
                         "severidade": worst})
        mlflow.log_param("intensidade_simulada", result["intensity"])
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"alerts": result["alerts"], "violations": result.get("violations", [])},
                        "alertas.json")
        if result.get("report"):
            mlflow.log_artifact(result["report"], artifact_path="drift")
    mlflow.log_metrics(metrics, step=step)


def totals(results):
    done = [r for r in results if r["status"] == "success"]
    flat = [alert for r in results for alert in r["alerts"]]
    return {
        "batches": len(results), "success": len(done),
        "rejected": sum(r["status"] == "rejected" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "rows_processed": sum(r["rows"] or 0 for r in done),
        "rows_blocked": sum(r["rows"] or 0 for r in results if r["status"] != "success"),
        "median_duration_s": median(r["duration_s"] for r in done) if done else None,
        "alerts": alert_rules.count_by_severity(flat),
    }


def run(manifest_path=config.MANIFEST):
    run_id = new_run_id("mon")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    model, model_dir = load_model()                      # modelo e métricas da mesma versão
    reference = score(model, pd.read_parquet(config.REFERENCE))
    baseline = load_baseline(reference, model_dir / config.BASELINE_METRICS.name)
    quarantine_dir = config.QUARANTINE_DIR / run_id      # identificador único por execução
    log_event("monitor", "started", rows=None, batches=len(manifest["batches"]))

    mlflow = tracking.start(config.EXPERIMENT_MONITORING)
    parent = tracking.safely(mlflow.start_run, run_name=run_id) if mlflow else None
    if parent:
        tracking.safely(mlflow.log_params, {
            "psi_aviso": config.PSI_WARNING, "psi_alerta": config.PSI_ALERT,
            "queda_auc": config.AUC_DROP, "queda_recall": config.RECALL_DROP,
            "baseline_auc": round(baseline["auc"], 4), "linhas_referencia": baseline["rows"],
            "versao_modelo": model_dir.name})

    results, durations = [], []
    try:
        for step, entry in enumerate(manifest["batches"], start=1):
            result = process_batch(entry, model=model, reference=reference, baseline=baseline,
                                   quarantine_dir=quarantine_dir, previous_durations=durations)
            if result["status"] == "success":
                durations.append(result["duration_s"])
            for alert in result["alerts"]:
                log_event("alert", alert["rule"], batch=alert["batch"],
                          level=EVENT_LEVEL[alert["severity"]],
                          **{k: v for k, v in alert.items() if k not in ("batch", "rule")})
            if parent:
                tracking.safely(track_batch, mlflow, result, step)
            results.append(result)
    finally:
        if parent:
            tracking.safely(mlflow.end_run)

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_dir.name,
        "features": list(model.feature_names_in_),
        "baseline": baseline,
        "thresholds": {"psi_warning": config.PSI_WARNING, "psi_alert": config.PSI_ALERT,
                       "auc_drop": config.AUC_DROP, "recall_drop": config.RECALL_DROP,
                       "default_rate_factor": config.DEFAULT_RATE_FACTOR,
                       "min_rows": config.MIN_ROWS, "latency_factor": config.LATENCY_FACTOR,
                       "decision_threshold": config.THRESHOLD},
        "simulation": {k: v for k, v in manifest.items() if k != "batches"},
        "totals": totals(results),
        "batches": results,
    }
    config.SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    config.SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_event("monitor", "success", rows=summary["totals"]["rows_processed"], **summary["totals"])
    return summary


def main():
    summary = run()
    print(f"\n{'lote':<28}{'status':<10}{'linhas':>7}{'AUC':>7}{'PSI máx':>9}  alertas")
    for r in summary["batches"]:
        auc = r.get("performance", {}).get("auc")
        psi_max = max((item["psi"] for item in r.get("drift", {}).values()), default=None)
        counts = alert_rules.count_by_severity(r["alerts"])
        flags = ", ".join(f"{count} {severity}" for severity, count in counts.items() if count) or "-"
        print(f"{r['batch']:<28}{r['status']:<10}{r['rows'] or 0:>7}"
              f"{'' if auc is None else format(auc, '.3f'):>7}"
              f"{'' if psi_max is None else format(psi_max, '.3f'):>9}  {flags}")
    print(f"\nResumo: {config.SUMMARY}")
    return 2 if summary["totals"]["failed"] else 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
