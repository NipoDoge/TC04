"""Parâmetros compartilhados do pipeline e do experimento sintético."""

SEED = 42
TARGET = "SeriousDlqin2yrs"
THRESHOLD = 0.5

# Fatores aplicados diretamente ao holdout original, sem acumulação.
PERIODS = (
    ("t0_control", 1.0, 1.0),
    ("t1", 0.90, 1.10),
    ("t2", 0.75, 1.30),
    ("t3", 0.60, 1.60),
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
