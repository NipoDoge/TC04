"""Parâmetros compartilhados do pipeline e do experimento sintético."""

SEED = 42
TARGET = "SeriousDlqin2yrs"
THRESHOLD = 0.5

# Cenários: nome, fator de utilização de crédito, fração com um atraso adicional.
# Intervenções aplicadas diretamente ao holdout original, sem acumulação.
UTILIZATION_COLUMN = "RevolvingUtilizationOfUnsecuredLines"
LATE90_COLUMN = "NumberOfTimes90DaysLate"
DATA_DRIFT_FEATURES = (UTILIZATION_COLUMN, LATE90_COLUMN)
PERIODS = (
    ("t0_control", 1.0, 0.0),
    ("t1", 1.10, 0.10),
    ("t2", 1.30, 0.25),
    ("t3", 1.60, 0.40),
)
SCENARIOS = tuple(name for name, _, _ in PERIODS)

# Tipagem de monitoramento explícita, sem inferência por cardinalidade.
DRIFT_CONTINUOUS_COLUMNS = (
    "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines",
)
DRIFT_KS_THRESHOLD = 0.01
DRIFT_PSI_THRESHOLD = 0.20
DRIFT_SHARE = 0.20

# Experimento de concept drift: choque adicional apenas no segmento de menor renda.
CONCEPT_INCOME_QUANTILE = 0.25
CONCEPT_PERIODS = (("c0_control", 0.0), ("c1", 0.10), ("c2", 0.25), ("c3", 0.50))
CONCEPT_SENSITIVITY_SEEDS = 5
CONCEPT_AUC_DROP = 0.05
CONCEPT_RECALL_DROP = 0.10
