import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pandera.pandas as pa

from src.contract import schema
from src.observability import log_event, new_run_id, setup_logging

OK, CONTRACT_REJECTED, OPERATIONAL_ERROR = 0, 1, 2
STATUS = {OK: "success", CONTRACT_REJECTED: "rejected", OPERATIONAL_ERROR: "failed"}
QUARANTINE = Path("data/quarantine")
log = logging.getLogger("ingest")


@dataclass
class IngestResult:
    code: int
    rows: int | None = None
    duration_s: float = 0.0
    violations: list = field(default_factory=list)   # agregados por regra, sem valores
    quarantine: str | None = None
    error: str | None = None


def summarize_violations(failure_cases):
    """Agrega as falhas por regra do contrato. Não carrega valores de clientes."""
    summary = []
    for (column, check), group in failure_cases.groupby(["column", "check"], dropna=False):
        summary.append({"column": None if pd.isna(column) else str(column), "check": str(check),
                        "rows": int(group["index"].nunique() or len(group))})
    # A unicidade da linha inteira é reportada uma vez por coluna: consolidar.
    duplicated = [item for item in summary if item["check"] == "multiple_fields_uniqueness"]
    if duplicated:
        summary = [item for item in summary if item not in duplicated]
        summary.append({"column": None, "check": "linhas duplicadas",
                        "rows": max(item["rows"] for item in duplicated)})
    return sorted(summary, key=lambda item: -item["rows"])


def validate_batch(path, quarantine_dir=QUARANTINE):
    """Valida um lote contra o contrato, coloca rejeitados em quarentena e
    registra o resultado no log central."""
    path, start = Path(path), time.perf_counter()

    def finish(result):
        result.duration_s = time.perf_counter() - start
        log_event("ingest", STATUS[result.code], batch=path.stem, rows=result.rows,
                  level="INFO" if result.code == OK else "ERROR", duration_s=result.duration_s,
                  violations=result.violations, quarantine=result.quarantine, error=result.error)
        return result

    try:
        df = pd.read_parquet(path)
    except Exception as error:
        log.error("batch=%s ERRO DE LEITURA: %s", path.name, error)
        return finish(IngestResult(OPERATIONAL_ERROR, error=f"leitura: {error}"))
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as error:
        qdir = Path(quarantine_dir)
        destination = qdir / path.name
        report = qdir / f"{path.stem}_errors.csv"
        try:
            qdir.mkdir(parents=True, exist_ok=True)
            if destination.exists() or report.exists():
                raise FileExistsError(f"Já existe evidência de quarentena para {path.name}")
            error.failure_cases.to_csv(report, index=False)
            shutil.move(path, destination)
        except Exception as failure:
            log.error("batch=%s ERRO NA QUARENTENA: %s", path.name, failure)
            return finish(IngestResult(OPERATIONAL_ERROR, rows=len(df), error=f"quarentena: {failure}"))
        log.error("batch=%s BLOQUEADO: %d falhas, movido para quarentena", path.name, len(error.failure_cases))
        return finish(IngestResult(CONTRACT_REJECTED, rows=len(df), quarantine=str(destination),
                                   violations=summarize_violations(error.failure_cases)))
    except Exception as error:
        log.error("batch=%s ERRO DE VALIDAÇÃO: %s", path.name, error)
        return finish(IngestResult(OPERATIONAL_ERROR, rows=len(df), error=f"validação: {error}"))
    log.info("batch=%s OK: %d linhas", path.name, len(df))
    return finish(IngestResult(OK, rows=len(df)))


def ingest(path, quarantine_dir=QUARANTINE):
    """Retorna 0 para aprovação, 1 para rejeição e 2 para falha operacional."""
    return validate_batch(path, quarantine_dir).code


def main(argv=None):
    parser = argparse.ArgumentParser(description="Valida um lote Parquet e bloqueia dados inválidos.")
    parser.add_argument("path", help="Caminho do lote Parquet")
    return ingest(parser.parse_args(argv).path)


if __name__ == "__main__":
    setup_logging()
    new_run_id("ingest")
    sys.exit(main())
