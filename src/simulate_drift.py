"""Cenários pareados de data drift; os períodos não são uma série temporal real."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import tempfile

import pandas as pd

from src.artifacts import sha256, snapshot
from src.config import PERIODS
from src.contract import schema

LIMITATION = (
    "O dataset não fornece uma série temporal real. Os períodos são sintéticos "
    "e reutilizam os mesmos clientes do holdout, na mesma ordem, para comparação "
    "pareada. Os rótulos históricos são preservados; o experimento não demonstra "
    "concept drift, causalidade econômica ou desempenho futuro real."
)


def build_scenarios(holdout):
    """Valida cópias sem modificar o holdout ou reaplicar filtros de outliers."""
    schema.validate(holdout, lazy=True)
    scenarios = {}
    for name, income_factor, debt_factor in PERIODS:
        period = holdout.copy(deep=True)
        period["MonthlyIncome"] *= income_factor
        period["DebtRatio"] *= debt_factor
        scenarios[name] = schema.validate(period, lazy=True)
    return scenarios


def simulate(holdout_path="data/production/holdout.parquet",
             output_dir="data/production/scenarios"):
    holdout_path, output_dir = Path(holdout_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    payload, source = snapshot(holdout_path)
    scenarios = build_scenarios(pd.read_parquet(BytesIO(payload)))
    manifest = {
        "limitation": LIMITATION,
        "source": source,
        "design": "paired_full_holdout",
        "sampling": "Todos os registros; sem amostragem ou aleatoriedade.",
        "target_policy": "SeriousDlqin2yrs preservado sem alterações.",
        "periods": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publicar o conjunto somente após validar e gravar todos os cenários.
    with tempfile.TemporaryDirectory(prefix=".scenarios-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "scenarios"
        staging.mkdir()
        for name, income_factor, debt_factor in PERIODS:
            path = staging / f"{name}.parquet"
            scenarios[name].to_parquet(path)
            manifest["periods"].append({
                "name": name, "file": path.name, "rows": len(scenarios[name]),
                "factors": {"MonthlyIncome": income_factor, "DebtRatio": debt_factor},
                "sha256": sha256(path),
            })
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", default="data/production/holdout.parquet")
    parser.add_argument("--output-dir", default="data/production/scenarios")
    args = parser.parse_args(argv)
    manifest = simulate(args.holdout, args.output_dir)
    print(LIMITATION)
    for period in manifest["periods"]:
        print(f"{period['name']}: {period['rows']} registros; fatores={period['factors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
