"""Demonstra falha, bloqueio, recuperação e drift em uma execução isolada."""

import argparse
from contextlib import chdir
import json
import os
from pathlib import Path
from uuid import uuid4

from src.concept_drift import run_experiment
from src.detect_drift import detect_drift
from src.ingest import ingest
from src.make_bad_batch import make_batches
from src.observability import run
from src.predict import predict
from src.prepare import prepare
from src.simulate_drift import simulate
from src.train import train


def demonstrate(raw_path="data/raw/cs-training.csv", output_dir="artifacts/operational_demo",
                execution_log="reports/observability/executions.jsonl"):
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Escolha um destino inexistente: {root}")
    raw_path = Path(raw_path).resolve()
    journal = Path(execution_log).resolve()
    root.mkdir(parents=True)
    run_id = f"demo-{uuid4().hex}"
    previous = {k: os.environ.get(k) for k in ("TC04_RUN_ID", "TC04_EXECUTION_LOG")}
    os.environ.update(TC04_RUN_ID=run_id, TC04_EXECUTION_LOG=str(journal))
    try:
        # Falhas reais e esperadas: o processo executa e os logs registram o resultado.
        try:
            prepare(root / "missing.csv", root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("A preparação deveria falhar com fonte ausente")
        prepare(raw_path, root)
        reference = root / "data/reference/reference.parquet"
        validation = root / "data/validation/validation.parquet"
        holdout = root / "data/production/holdout.parquet"
        train(reference, validation, root / "models")
        make_batches(holdout, root / "data/incoming")
        with chdir(root):
            if ingest("missing.parquet") != 2:
                raise AssertionError("Falha operacional não detectada")
            if ingest("data/incoming/bad_batch.parquet") != 1:
                raise AssertionError("Contrato deveria bloquear o lote inválido")
            if ingest("data/incoming/good_batch.parquet") != 0:
                raise AssertionError("Lote válido deveria ser aprovado após o bloqueio")
        scenarios = root / "data/production/scenarios"
        predictions = root / "data/predictions"
        drift = root / "reports/drift"
        concept = root / "concept_drift"
        model = root / "models/current/model.joblib"
        simulate(holdout, scenarios)
        predict(model, reference, scenarios, predictions)
        detect_drift(reference, scenarios, drift)
        run_experiment(reference, holdout, model, output_dir=concept)
        payload, paths = run(
            output_dir=root / "dashboard", execution_log=journal, run_id=run_id,
            drift_manifest=drift / "manifest.json", model_metrics=model.parent / "metrics.json",
            concept_manifest=concept / "manifest.json", quarantine_dir=root / "data/quarantine",
            predictions_manifest=predictions / "manifest.json", scenarios_manifest=scenarios / "manifest.json",
        )
        summary = {
            "run_id": run_id, "execution_log": str(journal),
            "pipeline": payload["metrics"]["pipeline"],
            "dashboard": str(paths["index_path"]),
            "note": "Falhas de preparação/ingestão e rejeição de contrato foram provocadas para demonstrar recuperação. Drift permanece como alerta esperado.",
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return summary
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/cs-training.csv")
    parser.add_argument("--output-dir", default="artifacts/operational_demo")
    parser.add_argument("--execution-log", default="reports/observability/executions.jsonl")
    args = parser.parse_args(argv)
    summary = demonstrate(args.raw, args.output_dir, args.execution_log)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
