"""Apoio aos testes: diretório temporário e dados sintéticos que respeitam o contrato."""
import logging
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.config import TARGET


def enter_temp_dir(case):
    """Os scripts usam caminhos relativos: cada teste roda em um diretório vazio."""
    temp = tempfile.TemporaryDirectory()
    previous = Path.cwd()
    os.chdir(temp.name)
    register = case.addClassCleanup if isinstance(case, type) else case.addCleanup
    register(temp.cleanup)
    register(os.chdir, previous)
    logger = logging.getLogger("pipeline")          # silencia o eco dos eventos no console
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    register(logger.setLevel, level)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        enter_temp_dir(self)


def synthetic_credit(n=3000, seed=0):
    """Clientes fictícios com as colunas e os tipos do contrato e um alvo que
    depende do uso do rotativo e dos atrasos (para o modelo ter o que aprender)."""
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
