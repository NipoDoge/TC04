"""Magnitude do drift e impacto no modelo nos cenários pareados da Etapa 2.

``src.detect_drift`` já responde se há drift em cada cenário (KS e PSI do
Evidently, por feature). Este script complementa aquela resposta com duas
medidas que o relatório não traz: o tamanho da mudança, pelo PSI com faixas nos
decis da referência (o mesmo do monitoramento por lote), e o efeito no modelo,
pelo PSI do score e pelas métricas de classificação. Como os cenários preservam
os rótulos históricos, uma queda de desempenho aqui mede só a sensibilidade do
modelo ao data drift; não representa concept drift.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from src import config
from src.batch_drift import load_model, score, status_for
from src.drift_stats import psi
from src.monitor import performance
from src.observability import new_run_id, setup_logging, track

NOTE = ("Cenários sintéticos pareados sobre o mesmo holdout, com rótulos históricos preservados. "
        "As métricas medem sensibilidade ao data drift; não demonstram concept drift nem desempenho futuro.")


def evidently_verdicts(report_dir):
    """Features apontadas pelo relatório do Evidently, se ``src.detect_drift`` já rodou."""
    manifest = Path(report_dir) / "manifest.json"
    if not manifest.exists():
        return {}
    scenarios = json.loads(manifest.read_text(encoding="utf-8")).get("scenarios", [])
    return {item["name"]: item["drifted_features"] for item in scenarios}


def run(scenarios_dir=config.SCENARIO_DIR, report_dir=config.SCENARIO_REPORTS,
        summary_path=config.SCENARIO_IMPACT, names=config.SCENARIOS):
    """Compara cada cenário com a referência e grava o resumo (somente agregados)."""
    scenarios_dir, report_dir, summary_path = Path(scenarios_dir), Path(report_dir), Path(summary_path)
    run_id = new_run_id("scn")
    model, model_dir = load_model()
    columns = list(model.feature_names_in_) + [config.SCORE]
    reference = score(model, pd.read_parquet(config.REFERENCE))
    factors = {name: {"MonthlyIncome": income, "DebtRatio": debt} for name, income, debt in config.PERIODS}
    verdicts = evidently_verdicts(report_dir)
    results = []
    for name in names:
        path = scenarios_dir / f"{name}.parquet"
        with track("scenario", batch=name) as info:
            scored = score(model, pd.read_parquet(path))
            drift = {}
            for column in columns:
                value = psi(reference[column], scored[column])
                drift[column] = {"psi": value, "status": status_for(value)}
            info.update(rows=len(scored), max_psi=max(item["psi"] for item in drift.values()),
                        alert=[c for c, item in drift.items() if item["status"] == "alert"],
                        warning=[c for c, item in drift.items() if item["status"] == "warning"])
        results.append({"scenario": name, "file": str(path), "rows": len(scored), "factors": factors.get(name),
                        "report": str(report_dir / f"{name}.html"), "evidently_drifted": verdicts.get(name),
                        "drift": drift, "performance": performance(scored)})
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_dir.name,
        "reference_rows": len(reference),
        "thresholds": {"psi_warning": config.PSI_WARNING, "psi_alert": config.PSI_ALERT,
                       "decision_threshold": config.THRESHOLD},
        "note": NOTE,
        "scenarios": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="PSI por decis e impacto no modelo nos cenários pareados.")
    parser.add_argument("--scenarios-dir", default=str(config.SCENARIO_DIR))
    parser.add_argument("--report-dir", default=str(config.SCENARIO_REPORTS),
                        help="Diretório dos relatórios de src.detect_drift (só leitura)")
    parser.add_argument("--summary", default=str(config.SCENARIO_IMPACT))
    args = parser.parse_args(argv)
    summary = run(args.scenarios_dir, args.report_dir, args.summary)
    print(f"{'cenário':<12}{'AUC':>7}{'recall':>8}{'sinaliz.':>10}{'PSI renda':>11}{'PSI dívida':>12}{'PSI score':>11}")
    for item in summary["scenarios"]:
        metrics, drift = item["performance"], item["drift"]
        print(f"{item['scenario']:<12}{metrics['auc']:>7.3f}{metrics['recall']:>8.3f}{metrics['flagged_rate']:>10.1%}"
              f"{drift['MonthlyIncome']['psi']:>11.3f}{drift['DebtRatio']['psi']:>12.3f}{drift[config.SCORE]['psi']:>11.3f}")
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
