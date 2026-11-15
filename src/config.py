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
