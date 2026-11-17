import argparse
import logging
from pathlib import Path
import shutil
import sys

import pandas as pd
import pandera.pandas as pa

from src.contract import schema
from src.execution_log import record, tracked

OK, CONTRACT_REJECTED, OPERATIONAL_ERROR = 0, 1, 2
log = logging.getLogger("ingest")


def summarize_violations(failure_cases):
    """Agrega por regra, sem carregar valores; consolida duplicatas entre colunas.

    Adaptado de feat/fase4-entrega-final (4f50a83), src/ingest.py.
    """
    summary = []
    for (column, check), group in failure_cases.groupby(["column", "check"], dropna=False):
        summary.append({"column": None if pd.isna(column) else str(column), "check": str(check),
                        "rows": int(group["index"].nunique() or len(group))})
    duplicated = [item for item in summary if item["check"] == "multiple_fields_uniqueness"]
    if duplicated:
        summary = [item for item in summary if item not in duplicated]
        summary.append({"column": None, "check": "linhas duplicadas",
                        "rows": max(item["rows"] for item in duplicated)})
    return sorted(summary, key=lambda item: -item["rows"])


@tracked("ingest", exit_codes=True)
def ingest(path, quarantine_dir="data/quarantine"):
    """Retorna 0 para aprovação, 1 para rejeição e 2 para falha operacional."""
    path = Path(path)
    try:
        df = pd.read_parquet(path)
        record(volume=len(df), input_rows=len(df))
    except Exception as error:
        record(error_type=type(error).__name__)
        log.error("batch=%s ERRO DE LEITURA: %s", path.name, error)
        return OPERATIONAL_ERROR
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as error:
        record(error_type=type(error).__name__, contract_failure_count=len(error.failure_cases),
               violations=summarize_violations(error.failure_cases))
        qdir = Path(quarantine_dir)
        destination = qdir / path.name
        report = qdir / f"{path.stem}_errors.csv"
        try:
            qdir.mkdir(parents=True, exist_ok=True)
            if destination.exists() or report.exists():
                raise FileExistsError(f"Já existe evidência de quarentena para {path.name}")
            error.failure_cases.to_csv(report, index=False)
            shutil.move(path, destination)
            record(quarantine=str(destination))
        except Exception as failure:
            record(error_type=type(failure).__name__)
            log.error("batch=%s ERRO NA QUARENTENA: %s", path.name, failure)
            return OPERATIONAL_ERROR
        log.error("batch=%s BLOQUEADO: %d falhas, movido para quarentena", path.name, len(error.failure_cases))
        return CONTRACT_REJECTED
    except Exception as error:
        record(error_type=type(error).__name__)
        log.error("batch=%s ERRO DE VALIDAÇÃO: %s", path.name, error)
        return OPERATIONAL_ERROR
    log.info("batch=%s OK: %d linhas", path.name, len(df))
    return OK


def main(argv=None):
    parser = argparse.ArgumentParser(description="Valida um lote Parquet e bloqueia dados inválidos.")
    parser.add_argument("path", help="Caminho do lote Parquet")
    return ingest(parser.parse_args(argv).path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
