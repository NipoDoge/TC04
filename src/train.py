import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

import joblib
import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from src.config import SEED, TARGET, THRESHOLD
from src.contract import FEATURE_COLUMNS, schema

log = logging.getLogger("train")


class TrainingDataError(ValueError):
    """Dados insuficientes para treinamento ou avaliação binária."""


def load_dataset(path):
    df = schema.validate(pd.read_parquet(path), lazy=True)
    if df.empty or df[TARGET].nunique() != 2:
        raise TrainingDataError(f"{path}: são necessários registros das duas classes")
    return df


def publish_artifacts(model, metrics, output_dir):
    """Grava uma versão completa e publica-a por troca atômica de current."""
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    published_dir = runs_dir / run_id
    pending_link = output_dir / f".current-{run_id}"

    # O temporário e o destino ficam no mesmo filesystem para permitir renames.
    with tempfile.TemporaryDirectory(prefix=".tmp-", dir=output_dir) as temporary:
        staging = Path(temporary)
        joblib.dump(model, staging / "model.joblib")
        (staging / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(published_dir)
        try:
            pending_link.symlink_to(Path("runs") / run_id, target_is_directory=True)
            os.replace(pending_link, output_dir / "current")
        except Exception:
            pending_link.unlink(missing_ok=True)
            shutil.rmtree(published_dir)
            raise
    return published_dir


def train(reference_path="data/reference/reference.parquet",
          validation_path="data/validation/validation.parquet", output_dir="models"):
    # Validar ambas as partições antes de criar o modelo ou seus artefatos.
    ref = load_dataset(reference_path)
    validation = load_dataset(validation_path)
    Xtr, ytr = ref.loc[:, FEATURE_COLUMNS], ref[TARGET]
    Xte, yte = validation.loc[:, FEATURE_COLUMNS], validation[TARGET]

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
        f"recall_at_{THRESHOLD:g}": float(recall_score(yte, p >= THRESHOLD)),
        "ks": float((tpr - fpr).max()),
    }
    publish_artifacts(model, metrics, output_dir)
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
