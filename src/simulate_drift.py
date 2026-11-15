"""Cenários pareados de data drift; os períodos não são uma série temporal real."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.artifacts import sha256, snapshot
from src.config import LATE90_COLUMN, PERIODS, SEED, UTILIZATION_COLUMN
from src.contract import schema

LIMITATION = (
    "O dataset não fornece uma série temporal real. Os períodos são sintéticos "
    "e reutilizam os mesmos clientes do holdout, na mesma ordem, para comparação "
    "pareada. Os rótulos históricos são preservados; o experimento não demonstra "
    "concept drift, causalidade econômica ou desempenho futuro real."
)


def build_scenarios(holdout, seed=SEED):
    """Valida cópias sem modificar o holdout ou reaplicar filtros de outliers."""
    schema.validate(holdout, lazy=True)
    # Os mesmos sorteios tornam os grupos de intervenção aninhados entre períodos.
    draws = np.random.default_rng(seed).random(len(holdout))
    scenarios = {}
    for name, utilization_factor, late_probability in PERIODS:
        period = holdout.copy(deep=True)
        period[UTILIZATION_COLUMN] *= utilization_factor
        period.loc[draws < late_probability, LATE90_COLUMN] += 1
        scenarios[name] = schema.validate(period, lazy=True)
    return scenarios


def simulate(holdout_path="data/production/holdout.parquet",
             output_dir="data/production/scenarios", seed=SEED):
    holdout_path, output_dir = Path(holdout_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    payload, source = snapshot(holdout_path)
    holdout = pd.read_parquet(BytesIO(payload))
    scenarios = build_scenarios(holdout, seed)
    manifest = {
        "limitation": LIMITATION,
        "source": source,
        "design": "paired_full_holdout",
        "scenario_definition": "credit_utilization_late90_v1",
        "seed": seed,
        "sampling": "Todos os registros; a composição da amostra permanece fixa.",
        "randomization": "U uniforme por linha, compartilhado entre períodos; acrescenta um atraso quando U < probabilidade, independentemente de rótulos e scores.",
        "target_policy": "SeriousDlqin2yrs preservado sem alterações.",
        "periods": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publicar o conjunto somente após validar e gravar todos os cenários.
    with tempfile.TemporaryDirectory(prefix=".scenarios-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "scenarios"
        staging.mkdir()
        for name, utilization_factor, late_probability in PERIODS:
            path = staging / f"{name}.parquet"
            scenarios[name].to_parquet(path)
            manifest["periods"].append({
                "name": name, "file": path.name, "rows": len(scenarios[name]),
                "interventions": {
                    UTILIZATION_COLUMN: {"operation": "multiply", "factor": utilization_factor},
                    LATE90_COLUMN: {"operation": "add", "increment": 1, "selection_probability": late_probability},
                },
                "changed_rows": {
                    column: int(scenarios[name][column].ne(holdout[column]).sum())
                    for column in [UTILIZATION_COLUMN, LATE90_COLUMN]
                },
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
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    manifest = simulate(args.holdout, args.output_dir, args.seed)
    print(LIMITATION)
    for period in manifest["periods"]:
        print(f"{period['name']}: {period['rows']} registros; intervenções={period['interventions']}; linhas alteradas={period['changed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
