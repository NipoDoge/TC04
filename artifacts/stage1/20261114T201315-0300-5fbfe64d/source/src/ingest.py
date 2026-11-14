import argparse
import logging
from pathlib import Path
import shutil
import sys

import pandas as pd
import pandera.pandas as pa

from src.contract import schema

OK, CONTRACT_REJECTED, OPERATIONAL_ERROR = 0, 1, 2
log = logging.getLogger("ingest")


def ingest(path):
    """Retorna 0 para aprovação, 1 para rejeição e 2 para falha operacional."""
    path = Path(path)
    try:
        df = pd.read_parquet(path)
    except Exception as error:
        log.error("batch=%s ERRO DE LEITURA: %s", path.name, error)
        return OPERATIONAL_ERROR
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as error:
        qdir = Path("data/quarantine")
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
            return OPERATIONAL_ERROR
        log.error("batch=%s BLOQUEADO: %d falhas, movido para quarentena", path.name, len(error.failure_cases))
        return CONTRACT_REJECTED
    except Exception as error:
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
