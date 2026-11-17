"""Executa o modelo salvo na referência e nos quatro cenários sintéticos."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from src.artifacts import sha256, snapshot
from src.config import SCENARIOS, TARGET, THRESHOLD
from src.contract import FEATURE_COLUMNS, schema
from src.execution_log import record, tracked


def predict_dataset(model, dataset):
    schema.validate(dataset, lazy=True)
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("As features do modelo salvo não correspondem a FEATURE_COLUMNS")
    if list(model.classes_) != [0, 1]:
        raise ValueError("O modelo deve possuir as classes [0, 1], nessa ordem")
    probabilities = np.asarray(model.predict_proba(dataset.loc[:, FEATURE_COLUMNS]))
    if (probabilities.shape != (len(dataset), 2)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Probabilidades inválidas retornadas pelo modelo")
    # A posição permite associar cada resultado ao arquivo de origem imutável.
    return pd.DataFrame({
        "row_position": np.arange(len(dataset)),
        TARGET: dataset[TARGET].to_numpy(),
        "probability_0": probabilities[:, 0],
        "probability_1": probabilities[:, 1],
        "predicted_class": (probabilities[:, 1] >= THRESHOLD).astype(int),
    })


@tracked("predict")
def predict(model_path="models/current/model.joblib",
            reference_path="data/reference/reference.parquet",
            scenarios_dir="data/production/scenarios", output_dir="data/predictions"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    # Resolve current uma única vez e utiliza o mesmo modelo para os cinco datasets.
    model_bytes, model_source = snapshot(model_path)
    model = joblib.load(BytesIO(model_bytes))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_source,
        "feature_columns": list(FEATURE_COLUMNS),
        "probability_classes": [0, 1],
        "classification_rule": f"predicted_class = int(probability_1 >= {THRESHOLD:g})",
        "threshold": THRESHOLD,
        "row_mapping": "row_position é a posição zero-based no arquivo de origem, sem reordenação.",
        "limitations": (
            "A referência foi usada no treino; suas predições não são avaliação independente. "
            "Os cenários são sintéticos e preservam rótulos históricos. "
            "As predições não comprovam concept drift ou desempenho futuro real."
        ),
        "datasets": [],
    }
    inputs = {"reference": Path(reference_path)}
    inputs.update({name: Path(scenarios_dir) / f"{name}.parquet" for name in SCENARIOS})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predictions-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "predictions"
        staging.mkdir()
        for name, path in inputs.items():
            payload, source = snapshot(path)
            predictions = predict_dataset(model, pd.read_parquet(BytesIO(payload)))
            destination = staging / f"{name}.parquet"
            predictions.to_parquet(destination, index=False)
            manifest["datasets"].append({
                "name": name, "source": source, "rows": len(predictions),
                "file": destination.name,
                "sha256": sha256(destination),
            })
            record(volume=sum(d["rows"] for d in manifest["datasets"]))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/current/model.joblib")
    parser.add_argument("--reference", default="data/reference/reference.parquet")
    parser.add_argument("--scenarios-dir", default="data/production/scenarios")
    parser.add_argument("--output-dir", default="data/predictions")
    args = parser.parse_args(argv)
    manifest = predict(args.model, args.reference, args.scenarios_dir, args.output_dir)
    print(f"Modelo: {manifest['model']['path']}")
    for dataset in manifest["datasets"]:
        print(f"{dataset['name']}: {dataset['rows']} predições salvas em {args.output_dir}/{dataset['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
