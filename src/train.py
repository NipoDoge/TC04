import json
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from src.contract import schema

SEED, TARGET = 42, "SeriousDlqin2yrs"
log = logging.getLogger("train")


class TrainingDataError(ValueError):
    """Dados insuficientes para treinamento ou avaliação binária."""


def load_dataset(path):
    df = schema.validate(pd.read_parquet(path), lazy=True)
    if df.empty or df[TARGET].nunique() != 2:
        raise TrainingDataError(f"{path}: são necessários registros das duas classes")
    return df


def train(reference_path="data/reference/reference.parquet",
          validation_path="data/validation/validation.parquet", output_dir="models"):
    # Validar ambas as partições antes de criar o modelo ou seus artefatos.
    ref = load_dataset(reference_path)
    validation = load_dataset(validation_path)
    Xtr, ytr = ref.drop(columns=[TARGET]), ref[TARGET]
    Xte, yte = validation.drop(columns=[TARGET]), validation[TARGET]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
        random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    metrics = {
        "auc": float(roc_auc_score(yte, p)),
        "recall_at_0.5": float(recall_score(yte, p >= 0.5)),
        "ks": float((tpr - fpr).max()),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    try:
        metrics = train()
    except (pa.errors.SchemaErrors, pa.errors.SchemaError, TrainingDataError) as error:
        log.error("Treinamento BLOQUEADO: %s", error)
        return 1
    except Exception as error:
        log.error("Falha operacional no treinamento: %s", error)
        return 2
    print(metrics)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
