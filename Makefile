PYTHON ?= python
.DEFAULT_GOAL := help
.PHONY: help all report prepare train demo-bad demo-good simulate-drift predict detect-drift concept-drift test \
        scenario-impact simulate-batches monitor causal dashboard mlflow-ui
.NOTPARALLEL:

help:
	@echo "Alvos: prepare, train, demo-bad, demo-good, simulate-drift, predict, detect-drift, concept-drift, test"
	@echo "       scenario-impact, simulate-batches, monitor, causal, dashboard, report, all, mlflow-ui"
	@echo "Sequência: prepare -> train -> simulate-drift -> predict -> detect-drift"
	@echo "Monitoramento e observabilidade: report = scenario-impact + monitor + causal + dashboard"
	@echo "Fluxo completo a partir do CSV bruto: all = prepare + train + report + test"
	@echo "Use PYTHON=.venv/bin/python para selecionar o ambiente."

# Fluxo completo. O treino publica uma nova versão em models/runs/.
all: prepare train report test

# Monitoramento e observabilidade com o modelo já publicado em models/current.
report: scenario-impact monitor causal dashboard

# Usa o holdout existente; não executa prepare nem altera a referência.
simulate-drift:
	$(PYTHON) -m src.simulate_drift

# Usa artefatos existentes; não prepara dados nem treina novamente.
predict:
	$(PYTHON) -m src.predict

detect-drift:
	$(PYTHON) -m src.detect_drift

concept-drift:
	$(PYTHON) -m src.concept_drift

prepare:
	$(PYTHON) -m src.prepare

train:
	$(PYTHON) -m src.train

test:
	$(PYTHON) -m unittest discover -s tests -v

demo-bad:
	$(PYTHON) -m src.make_bad_batch
	@status=0; $(PYTHON) -m src.ingest data/incoming/bad_batch.parquet || status=$$?; \
	if [ "$$status" -ne 1 ]; then \
		echo "Falha: esperada rejeição pelo contrato (1), recebido $$status" >&2; exit 1; \
	fi; \
	if [ -e data/incoming/bad_batch.parquet ] || \
	   [ ! -s data/quarantine/bad_batch.parquet ] || \
	   [ ! -s data/quarantine/bad_batch_errors.csv ]; then \
		echo "Falha: quarentena ou relatório não confirmados" >&2; exit 1; \
	fi

demo-good:
	$(PYTHON) -m src.ingest data/incoming/good_batch.parquet

# Impacto dos cenários pareados no modelo, lido de reports/drift e data/predictions (não recalcula).
scenario-impact:
	$(PYTHON) -m src.scenario_impact

# Lotes mensais nos níveis do calendário de src/config.py e uma entrega corrompida.
simulate-batches:
	$(PYTHON) -m src.simulate_batches

# A ingestão move a entrega corrompida para a quarentena: os lotes são recriados a cada execução.
monitor: simulate-batches
	$(PYTHON) -m src.monitor

causal:
	$(PYTHON) -m src.causal_analysis

dashboard:
	$(PYTHON) -m src.dashboard

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db
