"""Drift de um lote de produção contra a referência (treino), com o Evidently.

Complementa ``src.detect_drift``, que publica os relatórios dos cenários pareados
com KS e PSI nativo. Aqui o alvo é o monitoramento lote a lote, e a pergunta é de
magnitude: quanto a distribuição mudou e se o score mudou junto.

Para cada variável e para o score do modelo o relatório traz o PSI por decis da
referência (regra de alerta) e o teste Kolmogorov-Smirnov (evidência complementar,
muito sensível em amostras grandes). O HTML é o entregável visual; o resumo
devolvido alimenta alertas, MLflow e dashboard.
"""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import pandas as pd

from src import config
from src.drift_stats import ks, register_evidently_psi


def status_for(psi_value):
    if psi_value >= config.PSI_ALERT:
        return "alert"
    return "warning" if psi_value >= config.PSI_WARNING else "ok"


def drift_report(reference, current, columns, html_path=None):
    """Compara as distribuições de ``columns`` e devolve o resumo por coluna."""
    # Importação tardia: o Evidently é pesado e só esta função precisa dele.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import ValueDrift
        from evidently.presets import DataDriftPreset

    columns = list(columns)
    method = register_evidently_psi(config.PSI_ALERT)
    definition = DataDefinition(numerical_columns=columns)
    report = Report([
        DataDriftPreset(columns=columns, method=method, threshold=config.PSI_ALERT),
        *[ValueDrift(column=column, method="ks", threshold=config.KS_ALPHA) for column in columns],
    ])
    snapshot = report.run(Dataset.from_pandas(current[columns], data_definition=definition),
                          Dataset.from_pandas(reference[columns], data_definition=definition))
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    values = {}
    for metric in snapshot.dict()["metrics"]:
        settings = metric["config"]
        if settings.get("type", "").endswith("ValueDrift"):
            values.setdefault(settings["column"], {})[settings["method"]] = float(metric["value"])

    summary = {}
    for column in columns:
        statistic, _ = ks(reference[column], current[column])
        summary[column] = {
            "psi": values[column][method],
            "ks_stat": statistic,
            "ks_pvalue": values[column]["ks"],
            "status": status_for(values[column][method]),
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
    features = list(model.feature_names_in_)
    reference = score(model, pd.read_parquet(config.REFERENCE))
    current = score(model, pd.read_parquet(args.batch))
    html = args.html or config.BATCH_REPORTS / f"{Path(args.batch).stem}.html"
    summary = drift_report(reference, current, features + [config.SCORE], html)
    print(json.dumps({name: {"psi": round(item["psi"], 4), "ks_pvalue": item["ks_pvalue"],
                             "status": item["status"]} for name, item in summary.items()}, indent=2))
    print(f"Relatório: {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
