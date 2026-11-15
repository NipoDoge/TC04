"""Compara efeitos dos choques anteriores e atuais com o mesmo modelo e holdout."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src.artifacts import sha256, snapshot
from src.concept_drift import classification_metrics, performance_report
from src.config import DATA_DRIFT_FEATURES, SCENARIOS, SEED, TARGET, THRESHOLD
from src.contract import FEATURE_COLUMNS, schema
from src.predict import predict_dataset

PREVIOUS_FEATURES = ("MonthlyIncome", "DebtRatio")
LIMITATIONS = (
    "Comparação de sensibilidade a intervenções sintéticas, com rótulos históricos preservados. "
    "Não representa desempenho econômico futuro, concept drift ou efeito causal das features. "
    "Os períodos têm nomes correspondentes, mas as doses e unidades dos choques são diferentes; "
    "a comparação não isola a importância das variáveis. Ganho e permutação descrevem o modelo fixo "
    "no holdout original. Desvios padrão das permutações não são intervalos de confiança."
)


def evaluate_profile(model, scenarios_dir, predictions_dir, reference_scores, reference_classes):
    """Confere hashes, alvos e predições antes de calcular os efeitos pareados."""
    scenario_manifest = json.loads((Path(scenarios_dir) / "manifest.json").read_text())
    prediction_manifest = json.loads((Path(predictions_dir) / "manifest.json").read_text())
    datasets, predictions, results = {}, {}, []
    for name in SCENARIOS:
        period = next(p for p in scenario_manifest["periods"] if p["name"] == name)
        entry = next(p for p in prediction_manifest["datasets"] if p["name"] == name)
        data_path, prediction_path = Path(scenarios_dir) / period["file"], Path(predictions_dir) / entry["file"]
        if sha256(data_path) != period["sha256"] or period["sha256"] != entry["source"]["sha256"]:
            raise ValueError(f"Hash de origem divergente em {name}")
        if sha256(prediction_path) != entry["sha256"]:
            raise ValueError(f"Hash de predições divergente em {name}")
        data = schema.validate(pd.read_parquet(data_path), lazy=True)
        saved = pd.read_parquet(prediction_path)
        expected = predict_dataset(model, data)
        pd.testing.assert_frame_equal(saved, expected)
        datasets[name], predictions[name] = data, saved
        scores, classes = saved.probability_1.to_numpy(), saved.predicted_class.to_numpy()
        results.append({
            "scenario": name, **classification_metrics(data[TARGET].to_numpy(), scores, classes),
            "mean_absolute_score_change": float(np.abs(scores - reference_scores).mean()),
            "changed_predictions": int(np.count_nonzero(classes != reference_classes)),
            "changed_to_positive": int(np.count_nonzero((classes == 1) & (reference_classes == 0))),
            "changed_to_negative": int(np.count_nonzero((classes == 0) & (reference_classes == 1))),
        })
    return datasets, predictions, results


def measure_importance(model, holdout, repeats=10):
    X, y = holdout.loc[:, FEATURE_COLUMNS], holdout[TARGET]
    baseline_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    permuted = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=repeats, random_state=SEED, n_jobs=1)
    gain = model.get_booster().get_score(importance_type="gain")
    gain_total = sum(gain.values())
    rows = [{"feature": column, "normalized_gain": float(gain.get(column, 0) / gain_total),
             "auc_drop_mean": float(permuted.importances_mean[i]),
             "auc_drop_std": float(permuted.importances_std[i])} for i, column in enumerate(FEATURE_COLUMNS)]
    importance = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    importance["permutation_rank"] = np.arange(1, len(importance) + 1)
    rng = np.random.default_rng(SEED)
    orders = [rng.permutation(len(X)) for _ in range(repeats)]
    joint_rows = []
    for profile, columns in [("previous_income_debt", PREVIOUS_FEATURES), ("current_credit_late90", DATA_DRIFT_FEATURES)]:
        drops = []
        for order in orders:
            changed = X.copy()
            changed.loc[:, list(columns)] = X.iloc[order].loc[:, list(columns)].to_numpy()
            drops.append(baseline_auc - float(roc_auc_score(y, model.predict_proba(changed)[:, 1])))
        joint_rows.append({"profile": profile, "features": ",".join(columns),
                           "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops))})
    return importance, pd.DataFrame(joint_rows)


def write_analysis(path, metrics, importance, joint, drift_summary):
    selected = ["profile", "scenario", "auc", "recall", "precision", "mean_score", "predicted_positive_rate",
                "changed_predictions", "brier", "log_loss", "auc_drop", "recall_drop"]
    final = metrics[metrics.scenario == "t3"].set_index("profile")
    previous, current = final.loc["previous_income_debt"], final.loc["current_credit_late90"]
    narrative = (
        f"Em t3, o choque anterior resulta em AUC {previous.auc:.6f}, recall {previous.recall:.2%} "
        f"e {previous.predicted_positive_rate:.2%} de previsões positivas. O choque atual resulta em "
        f"AUC {current.auc:.6f}, recall {current.recall:.2%} e {current.predicted_positive_rate:.2%} "
        f"de previsões positivas. O número de classes previstas alteradas contra o controle é "
        f"{int(previous.changed_predictions)} no perfil anterior e {int(current.changed_predictions)} no atual."
    )
    path.write_text(f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Comparativo de intervenções de data drift</title>
<style>body{{font:16px system-ui,sans-serif;color:#172b4d;margin:32px;max-width:1600px}}p{{line-height:1.6;max-width:1150px}}
table{{border-collapse:collapse;font-size:14px;margin:20px 0}}th,td{{border:1px solid #cdd6df;padding:8px;text-align:right}}
th{{background:#eaf1f9}}.table-wrap{{overflow:auto}}a{{color:#145da0}}</style>
<h1>Comparativo: renda/dívida versus utilização de crédito/atrasos de 90 dias</h1>
<p>{LIMITATIONS}</p><h2>Desenho</h2>
<p>Mesmo holdout, modelo, ordem de linhas, alvo e limiar 0,5. Perfil anterior: renda ×0,90/0,75/0,60 e dívida ×1,10/1,30/1,60.
Perfil atual: utilização ×1,10/1,30/1,60; um atraso adicional em 10%/25%/40% das linhas, selecionadas com seed 42 e grupos aninhados.
Os sorteios não utilizam o alvo nem os scores. Os controles coincidem com o holdout original.</p>
<h2>Efeitos pareados no modelo</h2><p>{narrative}</p>
<div class="table-wrap">{metrics[selected].to_html(index=False,float_format=lambda x:f"{x:.6f}")}</div>
<p>auc_drop e recall_drop são controle menos cenário: valores positivos indicam queda.
Recall maior junto com crescimento de falsos positivos não implica melhoria geral. Com rótulos históricos mantidos após alterar X,
essas métricas descrevem a sensibilidade do modelo ao teste de estresse, sem estimar desfechos econômicos futuros.</p>
<p>Em t3, os falsos positivos passam de {int(previous.fp)} no perfil anterior para {int(current.fp)} no atual;
a precisão varia de {previous.precision:.2%} para {current.precision:.2%}. O Brier varia de {previous.brier:.6f} para
{current.brier:.6f} e o log loss de {previous.log_loss:.6f} para {current.log_loss:.6f} (menor é melhor para ambos).
O aumento de recall vem acompanhado de mais previsões positivas, incluindo mais erros entre os negativos históricos.</p>
<h2>Importância no holdout original</h2><div class="table-wrap">{importance.to_html(index=False,float_format=lambda x:f"{x:.6f}")}</div>
<p>Ganho médio dos splits normalizado entre features; não é proporção do risco explicado.
Permutação: queda média de AUC em dez embaralhamentos por feature, com seed 42.</p>
<h2>Permutação conjunta das duplas</h2><div class="table-wrap">{joint.to_html(index=False,float_format=lambda x:f"{x:.6f}")}</div>
<p>As duas colunas são embaralhadas com a mesma ordem, preservando a relação entre elas. As duas duplas usam as mesmas dez permutações.</p>
<h2>Resultados estatísticos de drift</h2><div class="table-wrap">{drift_summary.to_html(index=False,float_format=lambda x:f"{x:.6g}")}</div>
<p>KS usa p-valor &lt; 0,01; PSI usa valor &ge; 0,20. Métodos e limiares foram mantidos.
Um choque fraco pode alterar a distribuição sem ultrapassar o limiar escolhido. O sinal global exige 20% das features com drift;
uma única feature relevante pode mudar os scores antes de esse sinal global disparar.</p>
<h2>Gráficos nativos do Evidently</h2><p><a href="performance_previous_t3.html">Desempenho anterior em t3</a> ·
<a href="performance_current_t3.html">Desempenho atual em t3</a></p></html>''', encoding="utf-8")


def compare_data_drift(previous_dir="artifacts/data_drift_income_debt", output_dir="reports/drift_comparison"):
    previous_dir, output_dir = Path(previous_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    archive = json.loads((previous_dir / "archive.json").read_text())
    for entry in archive["files"]:
        if sha256(previous_dir / entry["archived_file"]) != entry["sha256"]:
            raise ValueError("O arquivo histórico não corresponde ao snapshot preservado")
    payload, model_source = snapshot("models/current/model.joblib")
    model = joblib.load(BytesIO(payload))
    payload, holdout_source = snapshot("data/production/holdout.parquet")
    holdout = schema.validate(pd.read_parquet(BytesIO(payload)), lazy=True)
    _, reference_source = snapshot("data/reference/reference.parquet")
    control = predict_dataset(model, holdout)
    baseline_auc = float(roc_auc_score(holdout[TARGET], control.probability_1))
    baseline_recall = classification_metrics(holdout[TARGET].to_numpy(), control.probability_1.to_numpy(), control.predicted_class.to_numpy())["recall"]
    profiles = {"previous_income_debt": (previous_dir / "scenarios", previous_dir / "predictions", previous_dir / "reports"),
                "current_credit_late90": (Path("data/production/scenarios"), Path("data/predictions"), Path("reports/drift"))}
    metrics_rows, drift_rows, profile_predictions, inputs = [], [], {}, []
    for profile, (scenario_dir, prediction_dir, report_dir) in profiles.items():
        scenario_manifest = json.loads((scenario_dir / "manifest.json").read_text())
        prediction_manifest = json.loads((prediction_dir / "manifest.json").read_text())
        report_manifest = json.loads((report_dir / "manifest.json").read_text())
        if scenario_manifest["source"]["sha256"] != holdout_source["sha256"] or prediction_manifest["model"]["sha256"] != model_source["sha256"]:
            raise ValueError("A comparação exige o mesmo holdout e o mesmo modelo nos dois perfis")
        if report_manifest["reference"]["sha256"] != reference_source["sha256"]:
            raise ValueError("Os relatórios precisam usar a mesma referência")
        for period in scenario_manifest["periods"]:
            report = next(p for p in report_manifest["scenarios"] if p["name"] == period["name"])
            if report["source"]["sha256"] != period["sha256"]:
                raise ValueError("O relatório de drift não corresponde ao cenário avaliado")
        datasets, predictions, rows = evaluate_profile(model, scenario_dir, prediction_dir, control.probability_1.to_numpy(), control.predicted_class.to_numpy())
        pd.testing.assert_frame_equal(datasets[SCENARIOS[0]], holdout)
        allowed = set(PREVIOUS_FEATURES if profile == "previous_income_debt" else DATA_DRIFT_FEATURES)
        for name, data in datasets.items():
            pd.testing.assert_frame_equal(data.loc[:, [c for c in data.columns if c not in allowed]], holdout.loc[:, [c for c in data.columns if c not in allowed]])
        summary_path = report_dir / report_manifest["summary"]["file"]
        if sha256(summary_path) != report_manifest["summary"]["sha256"]:
            raise ValueError("Resumo estatístico inconsistente com o manifesto")
        summary = pd.read_csv(summary_path)
        for row in rows:
            detected = summary[(summary.scenario == row["scenario"]) & summary.drift_detected].feature.tolist()
            row.update({"profile": profile, "auc_drop": baseline_auc - row["auc"], "recall_drop": baseline_recall - row["recall"], "drifted_features": ",".join(detected)})
        metrics_rows.extend(rows)
        drift_rows.append(summary.assign(profile=profile))
        profile_predictions[profile] = predictions
        inputs.extend({"path": str(p.resolve()), "sha256": sha256(p)} for p in [scenario_dir / "manifest.json", prediction_dir / "manifest.json", report_dir / "manifest.json"])
    importance, joint = measure_importance(model, holdout)
    metrics, drift_summary = pd.DataFrame(metrics_rows), pd.concat(drift_rows, ignore_index=True)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": model_source, "holdout": holdout_source, "reference": reference_source,
                "inputs": inputs, "baseline_auc": baseline_auc, "classification_threshold": THRESHOLD,
                "permutation_seed": SEED, "permutation_repeats": 10, "limitations": LIMITATIONS,
                "versions": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "evidently"]}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "comparison"
        staging.mkdir()
        for name, frame in [("metrics.csv", metrics), ("feature_importance.csv", importance), ("joint_importance.csv", joint), ("drift_summary.csv", drift_summary)]:
            frame.to_csv(staging / name, index=False)
        for label, profile in [("previous", "previous_income_debt"), ("current", "current_credit_late90")]:
            evaluation = performance_report(control, profile_predictions[profile]["t3"])
            evaluation.save_html(str(staging / f"performance_{label}_t3.html"))
            evaluation.save_json(str(staging / f"performance_{label}_t3.json"))
        write_analysis(staging / "analysis.html", metrics, importance, joint, drift_summary)
        manifest["artifacts"] = [{"file": p.name, "sha256": sha256(p)} for p in sorted(staging.iterdir())]
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", default="artifacts/data_drift_income_debt")
    parser.add_argument("--output-dir", default="reports/drift_comparison")
    args = parser.parse_args(argv)
    metrics = compare_data_drift(args.previous_dir, args.output_dir)
    print(metrics[["profile", "scenario", "auc", "recall", "predicted_positive_rate", "changed_predictions"]].to_string(index=False))
    print(f"Análise: {args.output_dir}/analysis.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
