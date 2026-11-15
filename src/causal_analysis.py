"""Análise causal por intervenção: o que derruba o modelo, P(X) ou P(Y|X)?

Os experimentos pareados medem cada mecanismo sozinho no holdout inteiro. Aqui os
dois níveis aplicados ao último lote mensal são ligados um de cada vez, aos mesmos
clientes e com o mesmo sorteio (seed). A diferença entre cenários é o efeito
daquela intervenção *dentro da simulação*, inclusive a interação entre as duas
mudanças. O intervalo de confiança vem de bootstrap pareado sobre os clientes.

Isso não identifica a causa de uma degradação real; mostra como o diagnóstico
seria conduzido e quanto cada mecanismo pesa no cenário construído.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.batch_drift import load_model, score
from src.drift_stats import psi
from src.monitor import performance
from src.observability import new_run_id, setup_logging, track
from src.simulate_batches import apply_levels, income_cutoff, split_months

SCENARIOS = {
    "controle": "Sem choque: os mesmos clientes, como chegaram",
    "data_drift": "Só P(X): variáveis alteradas, rótulos originais",
    "concept_drift": "Só P(Y|X): rótulos alterados, variáveis originais",
    "ambos": "Choque completo, igual ao último mês simulado",
}
EFFECTS = ("auc", "recall", "precision", "ks", "default_rate", "mean_score", "flagged_rate")


def paired_auc_interval(control, treated, draws=300, seed=config.SEED):
    """IC de 95% da diferença de AUC (tratado - controle), reamostrando clientes."""
    rng, n, deltas = np.random.default_rng(seed), len(control), []
    y0, p0 = control[config.TARGET].to_numpy(), control[config.SCORE].to_numpy()
    y1, p1 = treated[config.TARGET].to_numpy(), treated[config.SCORE].to_numpy()
    while len(deltas) < draws:
        rows = rng.integers(0, n, n)
        if len(np.unique(y0[rows])) == 2 and len(np.unique(y1[rows])) == 2:
            deltas.append(roc_auc_score(y1[rows], p1[rows]) - roc_auc_score(y0[rows], p0[rows]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def analyse(model, reference, raw, seed, data_level, concept_level, cutoff):
    features = list(model.feature_names_in_)

    def shock(**options):
        return apply_levels(raw, data_level, concept_level, cutoff, seed, **options)

    frames = {
        "controle": raw,
        "data_drift": shock(concept=False),
        "concept_drift": shock(columns=()),
        "ambos": shock(),
        **{f"do({feature})": shock(columns=(feature,), concept=False) for feature in config.DRIFT_SHIFTED_COLUMNS},
    }
    scored = {name: score(model, frame) for name, frame in frames.items()}
    control = performance(scored["controle"])
    results = {}
    for name, frame in scored.items():
        metrics = performance(frame)
        results[name] = {
            "description": SCENARIOS.get(name, f"Intervenção isolada em {name[3:-1]}, rótulos originais"),
            "metrics": metrics,
            "effect": {key: metrics[key] - control[key] for key in EFFECTS},
            "auc_effect_ci95": None if name == "controle" else paired_auc_interval(scored["controle"], frame),
            "psi_score": psi(reference[config.SCORE], frame[config.SCORE]),
            "drifted_features": [f for f in features
                                 if psi(reference[f], frame[f]) >= config.PSI_ALERT],
        }
    data, concept, both = (results[name]["effect"]["auc"] for name in ("data_drift", "concept_drift", "ambos"))
    return {
        "data_level": data_level, "concept_level": concept_level, "seed": seed, "rows": len(raw),
        "scenarios": {name: results[name] for name in SCENARIOS},
        "feature_interventions": {name[3:-1]: results[name] for name in results if name.startswith("do(")},
        "auc_decomposition": {"data_drift": data, "concept_drift": concept,
                              "interaction": both - data - concept, "total": both},
    }


def main():
    with track("causal") as info:
        model, model_dir = load_model()
        reference_data = pd.read_parquet(config.REFERENCE)
        reference = score(model, reference_data)
        month, data_level, concept_level = config.MONTHLY_SCHEDULE[-1]
        raw = split_months(pd.read_parquet(config.HOLDOUT))[-1]
        # Mesma seed do último mês simulado: o cenário "ambos" reproduz aquele lote.
        result = analyse(model, reference, raw, config.SEED + len(config.MONTHLY_SCHEDULE) - 1,
                         data_level, concept_level, income_cutoff(reference_data))
        result.update(month=month, model_version=model_dir.name)
        config.CAUSAL.parent.mkdir(parents=True, exist_ok=True)
        config.CAUSAL.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        info.update(rows=len(raw), **result["auc_decomposition"])
    print(f"{'cenário':<18}{'AUC':>7}{'ΔAUC':>8}  {'IC 95%':<20}{'recall':>7}{'sinalizados':>12}")
    for name, item in result["scenarios"].items():
        ci = item["auc_effect_ci95"]
        print(f"{name:<18}{item['metrics']['auc']:>7.3f}{item['effect']['auc']:>+8.3f}  "
              f"{'' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':<20}"
              f"{item['metrics']['recall']:>7.3f}{item['metrics']['flagged_rate']:>12.1%}")
    print(f"Decomposição da queda de AUC: {result['auc_decomposition']}")
    return 0


if __name__ == "__main__":
    setup_logging()
    new_run_id("causal")
    raise SystemExit(main())
