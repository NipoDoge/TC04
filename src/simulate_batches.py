"""Simula lotes mensais de produção com um choque econômico progressivo.

Complementa ``src.simulate_drift``, que gera cenários pareados sobre o holdout
inteiro (só data drift, com fatores fixos). Aqui o holdout vira uma sequência de
lotes, como chegariam em produção, com data drift, concept drift e uma entrega
corrompida: é a entrada do monitoramento (``src.monitor``).

O holdout (nunca visto no treino) é dividido em lotes mensais. Os três primeiros
meses ficam intactos (período de controle); a partir do quarto, a intensidade ``t``
do choque cresce até 1. Tudo é sintético, determinístico (seed) e registrado no
manifesto, para que a detecção seja comparada com intervenções conhecidas.

Mecanismo simulado, por cliente:
  * uma fração ``stress_share * t`` das famílias entra em estresse financeiro
    (variável latente, não observada pelo modelo);
  * data drift, P(X): renda cai (inflação para todos, perda de emprego nas
    famílias em estresse), o comprometimento de renda sobe, o uso do rotativo
    aumenta e surgem novos atrasos de 30-59 dias;
  * concept drift, P(Y|X): bons pagadores em estresse passam a inadimplir, e uma
    parcela pequena dos demais também, sem qualquer sinal nas variáveis.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.observability import new_run_id, setup_logging, track

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
INTENSITY = [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CORRUPTED_MONTH = "2026-05"          # mês que recebe uma primeira entrega inválida

INCOME, DEBT = "MonthlyIncome", "DebtRatio"
UTILIZATION, DELAYS = "RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse"
SHOCKED_FEATURES = (INCOME, DEBT, UTILIZATION, DELAYS)

PARAMS = {
    "stress_share": 0.50,              # famílias em estresse financeiro quando t = 1
    "income_erosion": 0.15,            # perda real de renda de todos quando t = 1
    "stress_income_range": [0.4, 0.8], # renda remanescente das famílias em estresse
    "interest_increase": 0.15,         # encarecimento das parcelas quando t = 1
    "utilization_range": [0.2, 0.9],   # fração do limite livre consumida sob estresse
    "delay_probability": 0.50,         # famílias em estresse com novo atraso de 30-59 dias
    "default_stress": 0.15,            # bons pagadores em estresse que inadimplem
    "default_silent": 0.02,            # demais bons pagadores que inadimplem quando t = 1
}


def draw_latent(n, t, seed, params=PARAMS):
    """Sorteia o estado latente dos clientes. A mesma seed gera o mesmo estado."""
    rng = np.random.default_rng(seed)
    stress = rng.random(n) < params["stress_share"] * t
    remaining = rng.uniform(*params["stress_income_range"], n)
    utilization = rng.uniform(*params["utilization_range"], n)
    delayed = rng.random(n) < params["delay_probability"]
    extra_delays = 1 + rng.poisson(0.3, n)
    return {
        "stress": stress,
        "income_factor": (1 - params["income_erosion"] * t) * np.where(stress, remaining, 1.0),
        "utilization_fraction": np.where(stress, utilization, 0.0),
        "new_delays": np.where(stress & delayed, extra_delays, 0),
        "default_draw": rng.random(n),
    }


def apply_shock(df, t, seed, features=SHOCKED_FEATURES, concept=True, params=PARAMS):
    """Aplica o choque de intensidade ``t``. ``features`` e ``concept`` permitem
    intervir em uma parte do mecanismo por vez (análise causal contrafactual)."""
    out = df.copy()
    if t <= 0:
        return out
    latent = draw_latent(len(out), t, seed, params)
    factor = latent["income_factor"]
    if INCOME in features:
        out[INCOME] = np.round(df[INCOME].to_numpy() * factor)
    if DEBT in features:
        # Parcela fixa sobre renda menor, mais juros. Sem renda declarada a coluna
        # guarda o valor absoluto da dívida, que só sofre o efeito dos juros.
        has_income = df[INCOME].to_numpy() > 0
        out[DEBT] = (df[DEBT].to_numpy() * (1 + params["interest_increase"] * t)
                     / np.where(has_income, factor, 1.0))
    if UTILIZATION in features:
        used = df[UTILIZATION].to_numpy()
        room = np.clip(1 - used, 0, None)          # quem já estourou o limite não muda
        out[UTILIZATION] = used + latent["utilization_fraction"] * room
    if DELAYS in features:
        out[DELAYS] = df[DELAYS].to_numpy() + latent["new_delays"]
    if concept:
        good = df[config.TARGET].to_numpy() == 0
        draw, stress = latent["default_draw"], latent["stress"]
        flip = good & np.where(stress, draw < params["default_stress"],
                               draw < params["default_silent"] * t)
        out.loc[flip, config.TARGET] = 1
    return out


def split_months(holdout, seed=config.SEED):
    """Embaralha o holdout e o reparte em fatias mensais de tamanho quase igual."""
    shuffled = holdout.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(shuffled)), len(MONTHS))
    return [shuffled.iloc[chunk].reset_index(drop=True) for chunk in chunks]


def corrupt(df, seed):
    """Imita uma falha de integração na origem: valores que o contrato precisa barrar."""
    rng, bad = np.random.default_rng(seed), df.copy()
    rows = rng.permutation(len(bad))
    bad.loc[rows[:120], INCOME] = np.nan            # renda não enviada
    bad.loc[rows[120:160], "age"] = 0               # idade com valor-padrão do sistema
    bad.loc[rows[160:180], DEBT] = -1.0             # código de erro no lugar da razão
    return pd.concat([bad, bad.iloc[rows[180:380]]], ignore_index=True)   # reenvio duplicado


def build_batches(holdout, seed=config.SEED):
    """Devolve os lotes na ordem de chegada: [(metadados, dataframe), ...]."""
    batches = []
    for index, (month, t, part) in enumerate(zip(MONTHS, INTENSITY, split_months(holdout, seed))):
        shocked = apply_shock(part, t, seed + index).drop_duplicates().reset_index(drop=True)
        if month == CORRUPTED_MONTH:
            bad = corrupt(shocked, seed)
            batches.append(({"batch": f"{month}-entrega-corrompida", "month": month,
                             "intensity": t, "kind": "corrupted", "rows": len(bad)}, bad))
        batches.append(({"batch": month, "month": month, "intensity": t,
                         "kind": "regular", "rows": len(shocked)}, shocked))
    return batches


def main(holdout_path=config.HOLDOUT, batch_dir=config.BATCH_DIR, seed=config.SEED):
    batch_dir = Path(batch_dir)
    with track("simulate") as info:
        batches = build_batches(pd.read_parquet(holdout_path), seed)
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
            "source": str(holdout_path), "seed": seed, "parameters": PARAMS,
            "shocked_features": list(SHOCKED_FEATURES), "batches": entries,
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
        print(f"{entry['batch']:<28} t={entry['intensity']:.1f} linhas={entry['rows']}")
