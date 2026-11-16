"""Dados sintéticos adaptados de feat/fase4-entrega-final (4f50a83).

Não usa registros do dataset de crédito versionado nem lê ou grava arquivos.
"""
import numpy as np
import pandas as pd

from src.config import TARGET


def synthetic_credit(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    utilization = rng.beta(1, 3, n)
    delays = rng.poisson(0.3, n)
    risk = 1 / (1 + np.exp(3 - 3 * utilization - 0.8 * delays))
    return pd.DataFrame({
        TARGET: (rng.random(n) < risk).astype(int),
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": rng.integers(21, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": delays,
        "DebtRatio": rng.gamma(2.0, 0.2, n),
        "MonthlyIncome": np.round(rng.lognormal(8.5, 0.5, n)),
        "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
        "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
        "NumberRealEstateLoansOrLines": rng.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
        "NumberOfDependents": rng.poisson(0.8, n).astype(float),
    })
