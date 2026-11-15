PYTHON ?= python
.DEFAULT_GOAL := help
.PHONY: help prepare train demo-bad demo-good simulate-drift predict detect-drift concept-drift test

help:
	@echo "Alvos: prepare, train, demo-bad, demo-good, simulate-drift, predict, detect-drift, concept-drift, test"
	@echo "Sequência: prepare -> train -> simulate-drift -> predict -> detect-drift"
	@echo "Use PYTHON=.venv/bin/python para selecionar o ambiente."

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
