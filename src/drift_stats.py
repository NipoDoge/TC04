"""Estatísticas de drift: PSI com faixas fixadas na referência e teste KS.

O PSI nativo do Evidently usa faixas de largura igual sobre referência + produção.
Em variáveis de cauda longa (renda, endividamento) quase toda a massa cai na
primeira faixa e o índice fica insensível. Aqui as faixas são os decis da
referência, prática usual em risco de crédito, e o teste é registrado no
Evidently com o nome ``psi_decis`` para aparecer nos relatórios HTML.
"""
import numpy as np
from scipy.stats import ks_2samp

PSI_METHOD = "psi_decis"


def bin_edges(reference, bins=10):
    """Faixas fixadas na referência: decis, ou um valor por faixa em contagens.

    As faixas das pontas são abertas: um valor de produção fora do intervalo visto
    na referência conta na primeira ou na última faixa. Mudanças dentro de uma
    mesma faixa não alteram o índice (limitação de qualquer PSI por faixas)."""
    values = np.unique(np.asarray(reference, dtype=float))
    if len(values) <= 2 * bins:        # baixa cardinalidade: cada valor é uma faixa
        inner = (values[:-1] + values[1:]) / 2
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))[1:-1]
    return np.concatenate([[-np.inf], inner, [np.inf]])


def psi(reference, current, bins=10, eps=1e-4):
    """Population Stability Index da produção em relação à referência."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI exige amostras não vazias")
    edges = bin_edges(reference, bins)
    expected = np.clip(np.histogram(reference, edges)[0] / len(reference), eps, None)
    actual = np.clip(np.histogram(current, edges)[0] / len(current), eps, None)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def ks(reference, current):
    """Teste Kolmogorov-Smirnov de duas amostras: (estatística D, p-valor)."""
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def register_evidently_psi(threshold=0.2):
    """Registra o PSI por decis como teste estatístico do Evidently."""
    from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
    from evidently.legacy.core import ColumnType

    def implementation(reference, current, feature_type, limit):
        value = psi(reference, current)
        return value, value >= limit

    register_stattest(
        StatTest(name=PSI_METHOD, display_name="PSI (decis da referência)",
                 allowed_feature_types=[ColumnType.Numerical], default_threshold=threshold),
        default_impl=implementation,
    )
    return PSI_METHOD
