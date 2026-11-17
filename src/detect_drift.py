"""Compara referência e cenários com Evidently e publica relatórios de data drift."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

# O pipeline gera relatórios locais sem telemetria de uso do Evidently.
os.environ.setdefault("DO_NOT_TRACK", "1")

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
import numpy as np
import pandas as pd

from src.artifacts import sha256, snapshot
from src.config import (
    DRIFT_CONTINUOUS_COLUMNS, DRIFT_KS_THRESHOLD, DRIFT_PSI_THRESHOLD,
    DRIFT_SHARE, SCENARIOS,
)
from src.contract import FEATURE_COLUMNS, schema
from src.execution_log import record, tracked


def compare_datasets(reference, current):
    """Monitora apenas features; alvo e resultados de inferência ficam de fora."""
    schema.validate(reference, lazy=True)
    schema.validate(current, lazy=True)
    definition = DataDefinition(
        numerical_columns=list(DRIFT_CONTINUOUS_COLUMNS),
        categorical_columns=[c for c in FEATURE_COLUMNS if c not in DRIFT_CONTINUOUS_COLUMNS],
    )
    report = Report([
        DataDriftPreset(
            columns=list(FEATURE_COLUMNS),
            num_method="ks", num_threshold=DRIFT_KS_THRESHOLD,
            cat_method="psi", cat_threshold=DRIFT_PSI_THRESHOLD,
            drift_share=DRIFT_SHARE,
        ),
    ], include_tests=True)
    return report.run(
        current_data=Dataset.from_pandas(current.loc[:, FEATURE_COLUMNS], data_definition=definition),
        reference_data=Dataset.from_pandas(reference.loc[:, FEATURE_COLUMNS], data_definition=definition),
    )


def summarize(evaluation, scenario):
    """Extrai os resultados da API 0.7.23 sem recalcular os testes estatísticos."""
    rows = []
    metrics = evaluation.dict()["metrics"]
    for metric in metrics:
        config = metric["config"]
        if config["type"] != "evidently:metric_v2:ValueDrift":
            continue
        score = float(metric["value"])
        method, threshold = config["method"], float(config["threshold"])
        if method not in {"ks", "psi"} or not np.isfinite(score):
            raise ValueError("Método ou resultado inesperado do Evidently")
        rows.append({
            "scenario": scenario, "feature": config["column"], "method": method,
            "score": score, "score_kind": "p_value" if method == "ks" else "psi",
            "threshold": threshold,
            "drift_detected": score < threshold if method == "ks" else score >= threshold,
        })
    if len(rows) != len(FEATURE_COLUMNS) or {r["feature"] for r in rows} != set(FEATURE_COLUMNS):
        raise ValueError("O Evidently não retornou exatamente as features esperadas")
    counts = [m["value"] for m in metrics if m["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount"]
    if len(counts) != 1 or int(counts[0]["count"]) != sum(r["drift_detected"] for r in rows):
        raise ValueError("Resumo de drift inconsistente com o resultado do Evidently")
    return rows


@tracked("detect_drift")
def detect_drift(reference_path="data/reference/reference.parquet",
                 scenarios_dir="data/production/scenarios", output_dir="reports/drift"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    payload, reference_source = snapshot(reference_path)
    reference = schema.validate(pd.read_parquet(BytesIO(payload)), lazy=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {p: version(p) for p in ["evidently", "pandas", "numpy", "scipy"]},
        "reference": reference_source,
        "reference_rows": len(reference),
        "feature_columns": list(FEATURE_COLUMNS),
        "settings": {
            "continuous_columns": list(DRIFT_CONTINUOUS_COLUMNS),
            "continuous_method": "ks", "ks_p_value_threshold": DRIFT_KS_THRESHOLD,
            "categorical_method": "psi", "psi_threshold": DRIFT_PSI_THRESHOLD,
            "dataset_drift_share_threshold": DRIFT_SHARE,
        },
        "limitations": (
            "Os períodos são sintéticos, sem série temporal real. O controle pode diferir "
            "da referência por amostragem. Data drift não comprova degradação de desempenho, "
            "concept drift ou causalidade. Alvo e predições não são monitorados neste relatório. "
            "Os limiares são critérios iniciais do experimento, sem ajuste de múltiplas comparações."
        ),
        "scenarios": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".drift-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "drift"
        staging.mkdir()
        all_rows = []
        for name in SCENARIOS:
            payload, source = snapshot(Path(scenarios_dir) / f"{name}.parquet")
            current = pd.read_parquet(BytesIO(payload))
            evaluation = compare_datasets(reference, current)
            rows = summarize(evaluation, name)
            all_rows.extend(rows)
            html_path, json_path = staging / f"{name}.html", staging / f"{name}.json"
            evaluation.save_html(str(html_path))
            evaluation.save_json(str(json_path))
            drifted = [r["feature"] for r in rows if r["drift_detected"]]
            share = len(drifted) / len(FEATURE_COLUMNS)
            manifest["scenarios"].append({
                "name": name, "source": source, "rows": len(current),
                "drifted_features": drifted, "drift_share": share,
                "dataset_drift_detected": share >= DRIFT_SHARE,
                "reports": [
                    {"file": p.name, "sha256": sha256(p)} for p in [html_path, json_path]
                ],
            })
        summary_path = staging / "summary.csv"
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        manifest["summary"] = {"file": summary_path.name, "sha256": sha256(summary_path)}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    record(volume=sum(s["rows"] for s in manifest["scenarios"]),
           alert_count=sum(s["dataset_drift_detected"] for s in manifest["scenarios"]),
           manifest_path=str(output_dir / "manifest.json"))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/reference/reference.parquet")
    parser.add_argument("--scenarios-dir", default="data/production/scenarios")
    parser.add_argument("--output-dir", default="reports/drift")
    args = parser.parse_args(argv)
    manifest = detect_drift(args.reference, args.scenarios_dir, args.output_dir)
    for scenario in manifest["scenarios"]:
        print(f"{scenario['name']}: drift em {scenario['drifted_features']}; relatório={args.output_dir}/{scenario['name']}.html")
    # Drift é um resultado do experimento; falhas operacionais propagam erro.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
