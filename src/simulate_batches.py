"""Simula lotes mensais de produção com os mecanismos dos experimentos pareados.

Os experimentos pareados aplicam um mecanismo de cada vez ao holdout inteiro:
``src.simulate_drift`` muda as variáveis (data drift) e ``src.concept_drift`` muda
o rótulo de um segmento (concept drift). Aqui o holdout é embaralhado e repartido
em meses, e cada mês recebe os níveis definidos em ``config.MONTHLY_SCHEDULE``.
Assim o monitoramento observa, lote a lote, as mesmas intervenções que os
experimentos medem isoladamente. Um dos meses recebe antes uma entrega
corrompida, para exercitar o contrato de dados.

Tudo é sintético, determinístico (seed) e registrado no manifesto.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.concept_drift import shift_target
from src.observability import new_run_id, setup_logging, track
from src.simulate_drift import shift_features

INCOME, DEBT, UTILIZATION, DELAYS = config.DRIFT_SHIFTED_COLUMNS


def income_cutoff(reference):
    """Corte do segmento de menor renda, ajustado somente na referência."""
    return float(reference[INCOME].quantile(config.CONCEPT_INCOME_QUANTILE))


def apply_levels(data, data_level, concept_level, cutoff, seed, columns=config.DRIFT_SHIFTED_COLUMNS, concept=True):
    """Aplica a um conjunto de clientes um nível de data drift e um de concept drift.

    ``columns`` e ``concept`` permitem ligar uma parte do choque por vez (análise
    contrafactual). O segmento do concept drift usa a renda original do cliente, e a
    mesma seed gera o mesmo sorteio em todos os cenários."""
    factors = {name: values for name, *values in config.PERIODS}[data_level]
    shifted = shift_features(data, *factors, columns=columns)
    probability = dict(config.CONCEPT_PERIODS)[concept_level] if concept else 0.0
    segment = data[INCOME].le(cutoff).to_numpy()
    draws = np.random.default_rng(seed).random(len(data))
    shifted, _ = shift_target(shifted, segment, draws, probability)
    return shifted


def split_months(holdout, seed=config.SEED, months=None):
    """Embaralha o holdout e o reparte em fatias mensais de tamanho quase igual."""
    months = len(config.MONTHLY_SCHEDULE) if months is None else months
    shuffled = holdout.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(shuffled)), months)
    return [shuffled.iloc[chunk].reset_index(drop=True) for chunk in chunks]


def corrupt(df, seed):
    """Imita uma falha de integração na origem: valores que o contrato precisa barrar."""
    rng, bad = np.random.default_rng(seed), df.copy()
    rows = rng.permutation(len(bad))
    bad.loc[rows[:120], INCOME] = np.nan            # renda não enviada
    bad.loc[rows[120:160], "age"] = 0               # idade com valor-padrão do sistema
    bad.loc[rows[160:180], DEBT] = -1.0             # código de erro no lugar da razão
    return pd.concat([bad, bad.iloc[rows[180:380]]], ignore_index=True)   # reenvio duplicado


def build_batches(holdout, cutoff, seed=config.SEED):
    """Devolve os lotes na ordem de chegada: [(metadados, dataframe), ...]."""
    schedule = config.MONTHLY_SCHEDULE
    batches = []
    for index, ((month, data_level, concept_level), part) in enumerate(zip(schedule, split_months(holdout, seed, len(schedule)))):
        shocked = apply_levels(part, data_level, concept_level, cutoff, seed + index)
        shocked = shocked.drop_duplicates().reset_index(drop=True)
        levels = {"month": month, "data_level": data_level, "concept_level": concept_level}
        if month == config.CORRUPTED_MONTH:
            bad = corrupt(shocked, seed)
            batches.append(({"batch": f"{month}-entrega-corrompida", **levels, "kind": "corrupted", "rows": len(bad)}, bad))
        batches.append(({"batch": month, **levels, "kind": "regular", "rows": len(shocked)}, shocked))
    return batches


def main(holdout_path=config.HOLDOUT, batch_dir=config.BATCH_DIR, seed=config.SEED, reference_path=config.REFERENCE):
    batch_dir = Path(batch_dir)
    with track("simulate") as info:
        cutoff = income_cutoff(pd.read_parquet(reference_path))
        batches = build_batches(pd.read_parquet(holdout_path), cutoff, seed)
        batch_dir.mkdir(parents=True, exist_ok=True)
        for old in batch_dir.glob("*.parquet"):
            old.unlink()
        entries = []
        for meta, df in batches:
            path = batch_dir / f"{meta['batch']}.parquet"
            df.to_parquet(path)
            entries.append({**meta, "file": str(path)})
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(holdout_path), "seed": seed,
            "schedule": [{"month": month, "data_level": data, "concept_level": concept}
                         for month, data, concept in config.MONTHLY_SCHEDULE],
            "data_drift_levels": {name: dict(zip((INCOME, DEBT, UTILIZATION), values)) for name, *values in config.PERIODS},
            "concept_drift_levels": dict(config.CONCEPT_PERIODS),
            "income_cutoff": cutoff,
            "shocked_features": list(config.DRIFT_SHIFTED_COLUMNS), "batches": entries,
        }
        (batch_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        info.update(rows=sum(meta["rows"] for meta, _ in batches), batches=len(batches))
    return manifest


if __name__ == "__main__":
    setup_logging()
    new_run_id("sim")
    result = main()
    for entry in result["batches"]:
        print(f"{entry['batch']:<28} dados={entry['data_level']:<11} rótulo={entry['concept_level']:<11} linhas={entry['rows']}")
