"""Parâmetros compartilhados do pipeline, do experimento sintético e do monitoramento.

Os caminhos são relativos à raiz do projeto, como nos demais scripts.
"""
from pathlib import Path

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

# Monitoramento por lote: lotes mensais simulados, drift, alertas e observabilidade.
SCORE, PREDICTION = "score", "prediction"

REFERENCE = Path("data/reference/reference.parquet")     # usado no treino
HOLDOUT = Path("data/production/holdout.parquet")        # nunca visto pelo modelo
SCENARIO_DIR = Path("data/production/scenarios")         # cenários pareados (src.simulate_drift)
BATCH_DIR = Path("data/production/batches")              # lotes mensais (src.simulate_batches)
MANIFEST = BATCH_DIR / "manifest.json"
SCORED_DIR = Path("data/production/scored")
QUARANTINE_DIR = Path("data/quarantine")

MODEL_DIR = Path("models/current")                       # apontador da versão publicada pelo treino
MODEL = MODEL_DIR / "model.joblib"
BASELINE_METRICS = MODEL_DIR / "metrics.json"
RECALL_KEY = f"recall_at_{THRESHOLD:g}"                  # nome da métrica gravada pelo treino

REPORTS = Path("reports")
SCENARIO_REPORTS = REPORTS / "drift"                     # relatórios dos cenários (src.detect_drift)
BATCH_REPORTS = REPORTS / "batches"                      # relatórios dos lotes mensais (src.monitor)
SCENARIO_IMPACT = REPORTS / "scenario_impact.json"
SUMMARY = REPORTS / "monitoring_summary.json"
CAUSAL = REPORTS / "causal_analysis.json"
DASHBOARD = REPORTS / "dashboard.html"

LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "pipeline_events.jsonl"
TEXT_LOG = LOG_DIR / "pipeline.log"

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_BASELINE = "tc04-baseline"
EXPERIMENT_MONITORING = "tc04-monitoramento"

# Regras de alerta do monitoramento (pontos de partida documentados no README, seção 7).
# O PSI por lote usa faixas nos decis da referência; o alerta reaproveita o limiar de PSI
# dos relatórios de cenários, e o KS, exibido como evidência, o mesmo p-valor.
PSI_WARNING, PSI_ALERT = 0.10, DRIFT_PSI_THRESHOLD
KS_ALPHA = DRIFT_KS_THRESHOLD
AUC_DROP, RECALL_DROP = CONCEPT_AUC_DROP, CONCEPT_RECALL_DROP   # mesmos critérios do experimento de concept drift
DEFAULT_RATE_FACTOR = 1.5            # inadimplência observada / inadimplência da referência
MIN_ROWS = 500
LATENCY_FACTOR = 2.0
