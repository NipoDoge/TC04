import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, TARGET
from src.observability import new_run_id, setup_logging, track

OUTLIER_COLUMNS = ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]

def clean(df):
    report, n = {}, len(df)
    def step(name, new):
        nonlocal n
        report[name] = n - len(new); n = len(new); return new

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = step("duplicadas", df.drop_duplicates())
    df = step("idade fora de 18-100", df[df["age"].between(18, 100)])
    df = step("renda/dependentes nulos", df.dropna(subset=["MonthlyIncome", "NumberOfDependents"]))
    df = step("renda negativa", df[df["MonthlyIncome"] >= 0])
    late = [c for c in df.columns if "PastDue" in c or "90DaysLate" in c]
    df = step("codigos 96/98 (placeholder)", df[~df[late].isin([96, 98]).any(axis=1)])
    return df.reset_index(drop=True), report


def filter_outliers(df, limits):
    report = {}
    for col, limit in limits.items():
        before = len(df)
        df = df[df[col] <= limit]
        report[f"outlier {col} > p99"] = before - len(df)
    return df.reset_index(drop=True), report


def prepare_datasets(raw):
    df, report = clean(raw)
    ref, holdout = train_test_split(df, test_size=0.3, stratify=df[TARGET], random_state=SEED)
    train, validation = train_test_split(
        ref, test_size=0.2, stratify=ref[TARGET], random_state=SEED
    )
    # Ambos os limites são ajustados no treino, antes de filtrar qualquer partição.
    limits = {col: float(train[col].quantile(0.99)) for col in OUTLIER_COLUMNS}
    datasets, outlier_report, sizes = {}, {}, {}
    for name, partition in [("train", train), ("validation", validation), ("holdout", holdout)]:
        datasets[name], outlier_report[name] = filter_outliers(partition, limits)
        sizes[name] = {"before_outliers": len(partition), "after_outliers": len(datasets[name])}
        if datasets[name].empty or datasets[name][TARGET].nunique() < 2:
            raise ValueError(f"Partição {name} precisa conter as duas classes após a limpeza")
    report["outliers"] = outlier_report
    report["partitions"] = sizes
    return datasets, limits, report


def main():
    with track("prepare") as info:
        datasets, limits, report = prepare_datasets(pd.read_csv("data/raw/cs-training.csv"))
        paths = {
            "train": Path("data/reference/reference.parquet"),
            "validation": Path("data/validation/validation.parquet"),
            "holdout": Path("data/production/holdout.parquet"),
        }
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            datasets[name].to_parquet(path)
        for filename, payload in [("outlier_limits.json", limits), ("cleaning_report.json", report)]:
            Path("data/reference", filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        info.update(rows=sum(len(partition) for partition in datasets.values()),
                    partitions={name: len(partition) for name, partition in datasets.items()})
    print(report)


if __name__ == "__main__":
    setup_logging()
    new_run_id("prepare")
    main()
