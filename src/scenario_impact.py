"""Impacto do data drift no modelo, nos cenários pareados da Etapa 2.

``src.detect_drift`` publica, por feature, se há drift e de que tamanho.
``src.predict`` publica as probabilidades do modelo em cada cenário. Este script
junta as duas evidências, sem recalcular nenhuma: lê o resumo de drift e as
predições e mede o que muda nas decisões (PSI do score, métricas de
classificação e parcela sinalizada). Como os cenários preservam os rótulos
históricos, o resultado mede a sensibilidade do modelo ao data drift; não
representa concept drift.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from src import config
from src.batch_drift import status_for
from src.drift_stats import psi
from src.monitor import performance
from src.observability import new_run_id, setup_logging, track

NOTE = ("Cenários sintéticos pareados sobre o mesmo holdout, com rótulos históricos preservados. "
        "As métricas medem sensibilidade ao data drift; não demonstram concept drift nem desempenho futuro.")
PREDICTIONS = Path("data/predictions")


def scored(path):
    """Predições publicadas por ``src.predict``, no formato usado pelas métricas."""
    data = pd.read_parquet(path)
    return pd.DataFrame({config.TARGET: data[config.TARGET], config.SCORE: data["probability_1"]})


def run(report_dir=config.SCENARIO_REPORTS, predictions_dir=PREDICTIONS,
        summary_path=config.SCENARIO_IMPACT, names=config.SCENARIOS):
    """Reúne drift por feature e impacto nas predições de cada cenário (somente agregados)."""
    report_dir, predictions_dir, summary_path = Path(report_dir), Path(predictions_dir), Path(summary_path)
    run_id = new_run_id("scn")
    drift = pd.read_csv(report_dir / "summary.csv")
    manifest = json.loads((predictions_dir / "manifest.json").read_text(encoding="utf-8"))
    reference = scored(predictions_dir / "reference.parquet")
    factors = {name: dict(zip(config.DRIFT_SHIFTED_COLUMNS, values)) for name, *values in config.PERIODS}
    results = []
    for name in names:
        with track("scenario", batch=name) as info:
            current = scored(predictions_dir / f"{name}.parquet")
            rows = drift[drift["scenario"] == name]
            features = {row.feature: {"psi": float(row.effect_psi), "status": status_for(row.effect_psi),
                                      "evidently_drift": bool(row.drift_detected)} for row in rows.itertuples()}
            score_psi = psi(reference[config.SCORE], current[config.SCORE])
            info.update(rows=len(current), score_psi=score_psi,
                        drifted=[feature for feature, item in features.items() if item["evidently_drift"]])
        results.append({
            "scenario": name, "rows": len(current), "factors": factors.get(name),
            "report": str(report_dir / f"{name}.html"),
            "evidently_drifted": [feature for feature, item in features.items() if item["evidently_drift"]],
            "drift": {**features, config.SCORE: {"psi": score_psi, "status": status_for(score_psi)}},
            "performance": performance(current),
        })
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": manifest["model"],
        "reference_rows": len(reference),
        "thresholds": {"psi_warning": config.PSI_WARNING, "psi_alert": config.PSI_ALERT,
                       "decision_threshold": config.THRESHOLD},
        "shifted_columns": list(config.DRIFT_SHIFTED_COLUMNS),
        "note": NOTE,
        "scenarios": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Impacto do data drift dos cenários pareados nas predições.")
    parser.add_argument("--report-dir", default=str(config.SCENARIO_REPORTS),
                        help="Diretório publicado por src.detect_drift (só leitura)")
    parser.add_argument("--predictions-dir", default=str(PREDICTIONS),
                        help="Diretório publicado por src.predict (só leitura)")
    parser.add_argument("--summary", default=str(config.SCENARIO_IMPACT))
    args = parser.parse_args(argv)
    summary = run(args.report_dir, args.predictions_dir, args.summary)
    print(f"{'cenário':<12}{'AUC':>7}{'recall':>8}{'sinaliz.':>10}{'PSI score':>11}  maior efeito nas variáveis")
    for item in summary["scenarios"]:
        metrics, drift = item["performance"], item["drift"]
        feature, largest = max(((c, d) for c, d in drift.items() if c != config.SCORE), key=lambda pair: pair[1]["psi"])
        print(f"{item['scenario']:<12}{metrics['auc']:>7.3f}{metrics['recall']:>8.3f}{metrics['flagged_rate']:>10.1%}"
              f"{drift[config.SCORE]['psi']:>11.3f}  {feature} (PSI {largest['psi']:.3f})")
    print(f"Resumo: {args.summary}")
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
