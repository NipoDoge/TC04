PYTHON ?= python
.PHONY: prepare train demo-bad demo-good

prepare:
	$(PYTHON) -m src.prepare
train: prepare
	$(PYTHON) -m src.train
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
