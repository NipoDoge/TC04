"""Drift de um lote de produção contra a referência (treino).

Usa o mesmo relatório do Evidently dos cenários pareados
(``src.detect_drift.compare_datasets``: KS nas medidas contínuas e PSI de
categorias nas demais), de modo que cenários e lotes são lidos com o mesmo
critério. Acrescenta o tamanho da mudança de cada variável e do score do modelo
(``src.drift_stats``), que é o que as regras de alerta do monitoramento usam.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.contract import FEATURE_COLUMNS, schema
from src.detect_drift import compare_datasets, summarize
from src.drift_stats import ks, psi

CONTRACT_COLUMNS = list(schema.columns)


def status_for(psi_value):
    if psi_value >= config.PSI_ALERT:
        return "alert"
    return "warning" if psi_value >= config.PSI_WARNING else "ok"


def drift_report(reference, current, html_path=None, name="lote"):
    """Compara as distribuições e devolve o resumo por coluna (features e score).

    ``reference`` e ``current`` trazem as colunas do contrato e o score. O Evidently
    recebe só as colunas do contrato; o score entra pelo tamanho do efeito."""
    evaluation = compare_datasets(reference.loc[:, CONTRACT_COLUMNS], current.loc[:, CONTRACT_COLUMNS])
    verdicts = {row["feature"]: row["drift_detected"] for row in summarize(evaluation, name)}
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation.save_html(str(html_path))

    summary = {}
    for column in [*FEATURE_COLUMNS, config.SCORE]:
        statistic, pvalue = ks(reference[column], current[column])
        effect = psi(reference[column], current[column])
        summary[column] = {
            "psi": effect, "status": status_for(effect),
            "ks_stat": statistic, "ks_pvalue": pvalue,
            "evidently_drift": verdicts.get(column),          # None para o score: fora do relatório
            "ref_mean": float(reference[column].mean()), "cur_mean": float(current[column].mean()),
            "ref_median": float(reference[column].median()), "cur_median": float(current[column].median()),
        }
    return summary


def load_model(model_dir=None):
    """Resolve o apontador ``models/current`` uma única vez e carrega o modelo dessa
    versão; as métricas do baseline devem ser lidas do mesmo diretório devolvido."""
    run_dir = Path(model_dir or config.MODEL_DIR).resolve()
    return joblib.load(run_dir / config.MODEL.name), run_dir


def score(model, df):
    """Acrescenta a probabilidade de inadimplência e a classe prevista."""
    scores = model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
    return df.assign(**{config.SCORE: scores,
                        config.PREDICTION: (scores >= config.THRESHOLD).astype(int)})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Relatório de drift de um lote contra a referência.")
    parser.add_argument("batch", help="Lote Parquet de produção")
    parser.add_argument("--html", default=None, help="Destino do relatório HTML do Evidently")
    args = parser.parse_args(argv)
    model, _ = load_model()
    reference = score(model, pd.read_parquet(config.REFERENCE))
    current = score(model, pd.read_parquet(args.batch))
    html = args.html or config.BATCH_REPORTS / f"{Path(args.batch).stem}.html"
    summary = drift_report(reference, current, html, Path(args.batch).stem)
    print(json.dumps({name: {"psi": round(item["psi"], 4), "ks_pvalue": item["ks_pvalue"],
                             "evidently_drift": item["evidently_drift"], "status": item["status"]}
                      for name, item in summary.items()}, indent=2))
    print(f"Relatório: {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
