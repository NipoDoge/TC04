"""Gera lotes de demonstração do contrato da Etapa 1, sem simular drift."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract import schema


def make_batches(holdout_path="data/production/holdout.parquet",
                 output_dir="data/incoming"):
    holdout = schema.validate(pd.read_parquet(holdout_path), lazy=True)
    if len(holdout) < 500:
        raise ValueError("O holdout precisa conter pelo menos 500 registros")
    bad = holdout.sample(500, random_state=1).reset_index(drop=True)
    bad.loc[0, "age"] = 15
    bad.loc[1, "MonthlyIncome"] = np.nan
    bad.loc[2, "DebtRatio"] = -1.0
    bad = pd.concat([bad, bad.iloc[3:8]], ignore_index=True)
    bad = bad.drop(columns="NumberOfDependents")
    good = holdout.sample(500, random_state=2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bad.to_parquet(output_dir / "bad_batch.parquet")
    good.to_parquet(output_dir / "good_batch.parquet")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", default="data/production/holdout.parquet")
    parser.add_argument("--output-dir", default="data/incoming")
    args = parser.parse_args(argv)
    make_batches(args.holdout, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
