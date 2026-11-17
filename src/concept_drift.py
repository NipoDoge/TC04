"""Experimento pareado de mudança sintética em P(Y|X), com X e modelo fixos."""

import argparse
from datetime import datetime, timezone
from html import escape
from importlib.metadata import version
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, roc_auc_score, roc_curve,
)

# Importar primeiro o módulo que desabilita a telemetria do Evidently.
from src.detect_drift import compare_datasets, summarize
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset

from src.artifacts import sha256, snapshot
from src.config import (
    CONCEPT_AUC_DROP, CONCEPT_INCOME_QUANTILE, CONCEPT_PERIODS,
    CONCEPT_RECALL_DROP, CONCEPT_SENSITIVITY_SEEDS, SEED, TARGET, THRESHOLD,
)
from src.contract import FEATURE_COLUMNS, schema
from src.predict import predict_dataset
from src.execution_log import record, tracked

LIMITATIONS = (
    "Experimento sintético, sem série temporal real ou inferência de causa econômica. "
    "O mecanismo de alteração dos rótulos é uma hipótese controlada, não desfechos observados. "
    "As métricas comparam o mesmo holdout com o seu controle histórico, não com a validação. "
    "Nenhum parâmetro do modelo ou limiar de classificação é ajustado neste experimento. "
    "As cinco seeds avaliam sensibilidade aos sorteios; não são intervalos de confiança. "
    "Na operação real, métricas supervisionadas exigem rótulos maturados no horizonte do alvo."
)


def performance_degraded(auc_drop, recall_drop):
    """Aplica os limiares inclusive no limite, tolerando arredondamento binário."""
    return any(value is not None and (value >= threshold or math.isclose(value, threshold, abs_tol=1e-12, rel_tol=0))
               for value, threshold in ((auc_drop, CONCEPT_AUC_DROP), (recall_drop, CONCEPT_RECALL_DROP)))


def build_concept_scenarios(holdout, income_cutoff, seed=SEED):
    """Yq = Y0 OR (income <= cutoff AND U < q); não utiliza scores do modelo."""
    schema.validate(holdout, lazy=True)
    if not np.isfinite(income_cutoff) or income_cutoff < 0:
        raise ValueError("O corte de renda precisa ser finito e não negativo")
    segment = holdout["MonthlyIncome"].le(income_cutoff).to_numpy()
    if not segment.any() or segment.all():
        raise ValueError("São necessários registros dentro e fora do segmento de renda")
    original = holdout[TARGET].to_numpy()
    if np.unique(original).size != 2:
        raise ValueError("O holdout precisa conter as duas classes")
    draws = np.random.default_rng(seed).random(len(holdout))
    audit = pd.DataFrame({
        "row_position": np.arange(len(holdout)), "original_target": original,
        "income_segment": segment, "uniform_draw": draws,
    })
    scenarios = {}
    for name, probability in CONCEPT_PERIODS:
        current = holdout.copy(deep=True)
        changed = segment & (original == 0) & (draws < probability)
        current.loc[changed, TARGET] = 1
        scenarios[name] = schema.validate(current, lazy=True)
        if current[TARGET].nunique() != 2:
            raise ValueError(f"O cenário {name} precisa conter as duas classes para avaliar discriminação")
        audit[f"{name}_changed"] = changed
    return scenarios, audit


def classification_metrics(labels, probabilities, predicted):
    """Métricas no limiar fixo; discriminação indefinida retorna None em segmento de uma classe."""
    labels = np.asarray(labels)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    has_two_classes = np.unique(labels).size == 2
    fpr, tpr, _ = roc_curve(labels, probabilities) if has_two_classes else (None, None, None)
    return {
        "rows": len(labels), "positive_labels": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "mean_score": float(np.mean(probabilities)),
        "predicted_positive_rate": float(np.mean(predicted)),
        "auc": float(roc_auc_score(labels, probabilities)) if has_two_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if has_two_classes else None,
        "ks_classification": float((tpr - fpr).max()) if has_two_classes else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "accuracy": (tp + tn) / len(labels),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "brier": float(brier_score_loss(labels, probabilities, pos_label=1)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def performance_report(control, current):
    definition = DataDefinition(classification=[BinaryClassification(
        target=TARGET, prediction_labels="predicted_class", prediction_probas="probability_1",
        pos_label=1,
    )])
    columns = [TARGET, "predicted_class", "probability_1"]
    return Report([ClassificationPreset(probas_threshold=THRESHOLD)], include_tests=False).run(
        current_data=Dataset.from_pandas(current.loc[:, columns], data_definition=definition),
        reference_data=Dataset.from_pandas(control.loc[:, columns], data_definition=definition),
    )


def write_analysis(path, results, segments, sensitivity, cutoff):
    """Relatório de leitura com tabelas e links para os gráficos nativos do Evidently."""
    overall = pd.DataFrame(results)
    selected = ["scenario", "changed_labels", "prevalence", "auc", "recall", "precision",
                "brier", "log_loss", "fn", "auc_drop", "recall_drop", "performance_alert"]
    links = " ".join(f'<a href="classification_{escape(r["scenario"])}.html">{escape(r["scenario"])}</a>' for r in results)
    grouped = sensitivity.groupby("scenario")[["auc", "recall"]].agg(["min", "max"])
    baseline, last = results[0], results[-1]
    interpretation = (
        f"No cenário {last['scenario']}, a AUC passa de {baseline['auc']:.6f} para {last['auc']:.6f} "
        f"e o recall de {baseline['recall']:.2%} para {last['recall']:.2%}. "
        f"Os falsos negativos passam de {baseline['fn']} para {last['fn']}. "
        f"O Brier varia de {baseline['brier']:.6f} para {last['brier']:.6f} e o log loss "
        f"de {baseline['log_loss']:.6f} para {last['log_loss']:.6f} (menor é melhor para ambos). "
        f"A precisão passa de {baseline['precision']:.2%} para {last['precision']:.2%}, "
        "sem qualquer adaptação do modelo: com a mesma lista de previsões positivas, "
        "a nova prevalência muda quantas delas são consideradas corretas. "
        "Por isso, uma melhora isolada de precisão ou F1 não implica recuperação da discriminação."
    )
    path.write_text(f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Concept drift: comportamento do modelo</title>
<style>body{{font:16px system-ui,sans-serif;color:#172b4d;margin:32px;max-width:1500px}}
table{{border-collapse:collapse;margin:20px 0;font-size:14px}}th,td{{border:1px solid #ccd5df;padding:8px;text-align:right}}
th{{background:#eaf1f9}}a{{color:#145da0;margin-right:12px}}.table-wrap{{overflow:auto}}p{{max-width:1100px;line-height:1.6}}</style>
<h1>Concept drift: X e modelo fixos, mecanismo do alvo alterado</h1>
<p>{escape(LIMITATIONS)}</p>
<h2>Desenho do experimento</h2><p>Segmento: MonthlyIncome &le; {cutoff:g}, primeiro quartil ajustado somente na referência.
Um sorteio U uniforme por cliente é compartilhado entre os períodos. Apenas rótulos 0 no segmento podem virar 1,
com probabilidades adicionais de 10%, 25% e 50%. Rótulos positivos são preservados.
Logo, Pq(Y=1|X)=p0(X)+(1-p0(X))*q*I(segmento), com p0 desconhecido, não estimado pelo modelo.</p>
<h2>Resultado global, limiar de classificação 0,5</h2><div class="table-wrap">{overall[selected].to_html(index=False, float_format=lambda v: f"{v:.6f}")}</div>
<p>{escape(interpretation)}</p>
<p>AUC e recall são comparados ao controle no mesmo holdout. Alerta experimental: queda absoluta de AUC &ge; {CONCEPT_AUC_DROP:g}
ou recall &ge; {CONCEPT_RECALL_DROP:g}. As taxas previstas e os scores são idênticos em todos os cenários;
o modelo não observa a mudança no mecanismo do alvo. A referência de treino não é usada como avaliação.</p>
<h2>Segmentos</h2><div class="table-wrap">{pd.DataFrame(segments)[['scenario','segment','rows','prevalence','auc','recall','precision','brier','fn']].to_html(index=False, float_format=lambda v: f"{v:.6f}")}</div>
<h2>Sensibilidade aos sorteios: cinco seeds</h2><div class="table-wrap">{grouped.to_html(float_format=lambda v: f"{v:.6f}")}</div>
<p>Os intervalos acima são mínimos/máximos entre seeds, sem interpretação de intervalo de confiança.
Precisão, average precision, Brier e log loss também respondem à prevalência: podem melhorar mesmo com piora de AUC/recall.
O treinamento com ponderação de classes não garante probabilidades calibradas. Todas as métricas e matrizes de confusão estão nos CSVs.</p>
<h2>Evidências e interpretação</h2><p>{links}<a href="feature_drift.html">Drift das features: c3 vs controle</a></p>
<p>Os relatórios de classificação mostram ROC, PR e matrizes de confusão. A igualdade exata das features e das predições é verificada
em todos os períodos; os testes de features do Evidently comparam c3 ao controle, e não ao treino.</p>
<p>O experimento ilustra uma degradação que monitoramento apenas de features e scores não detectaria.
Em operação, investigar origem e maturação dos rótulos, revisar erros por segmento e avaliar recalibração ou retreinamento em dados independentes.
O experimento não seleciona nem publica um modelo novo. Não autoriza conclusões sobre clientes reais ou causas econômicas.</p>
</html>''', encoding="utf-8")


@tracked("concept_drift")
def run_experiment(reference_path="data/reference/reference.parquet",
                   holdout_path="data/production/holdout.parquet",
                   model_path="models/current/model.joblib", seed=SEED,
                   output_dir="artifacts/concept_drift"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Destino já existe; escolha outro --output-dir: {output_dir}")
    payload, reference_source = snapshot(reference_path)
    reference = schema.validate(pd.read_parquet(BytesIO(payload)), lazy=True)
    cutoff = float(reference["MonthlyIncome"].quantile(CONCEPT_INCOME_QUANTILE))
    payload, holdout_source = snapshot(holdout_path)
    holdout = schema.validate(pd.read_parquet(BytesIO(payload)), lazy=True)
    scenarios, audit = build_concept_scenarios(holdout, cutoff, seed)
    payload, model_source = snapshot(model_path)
    model = joblib.load(BytesIO(payload))
    predictions = {name: predict_dataset(model, data) for name, data in scenarios.items()}
    control_name = CONCEPT_PERIODS[0][0]
    control = predictions[control_name]
    base_scores = control["probability_1"].to_numpy()
    base_classes = control["predicted_class"].to_numpy()
    results, segments = [], []
    segment = audit["income_segment"].to_numpy()
    for name, probability in CONCEPT_PERIODS:
        data, prediction = scenarios[name], predictions[name]
        pd.testing.assert_frame_equal(data.loc[:, FEATURE_COLUMNS], holdout.loc[:, FEATURE_COLUMNS])
        np.testing.assert_array_equal(prediction[["probability_0", "probability_1", "predicted_class"]],
                                      control[["probability_0", "probability_1", "predicted_class"]])
        labels = data[TARGET].to_numpy()
        row = {"scenario": name, "shock_probability": probability,
               "changed_labels": int(audit[f"{name}_changed"].sum()),
               **classification_metrics(labels, base_scores, base_classes)}
        baseline = results[0] if results else row
        row["auc_drop"] = baseline["auc"] - row["auc"]
        row["recall_drop"] = baseline["recall"] - row["recall"]
        row["performance_alert"] = performance_degraded(row["auc_drop"], row["recall_drop"])
        results.append(row)
        for label, mask in [("income_le_q25", segment), ("income_gt_q25", ~segment)]:
            segments.append({"scenario": name, "segment": label,
                             **classification_metrics(labels[mask], base_scores[mask], base_classes[mask])})
    sensitivity_rows = []
    for trial_seed in range(seed, seed + CONCEPT_SENSITIVITY_SEEDS):
        trial, _ = build_concept_scenarios(holdout, cutoff, trial_seed)
        for name, probability in CONCEPT_PERIODS:
            sensitivity_rows.append({"seed": trial_seed, "scenario": name, "shock_probability": probability,
                                     **classification_metrics(trial[name][TARGET].to_numpy(), base_scores, base_classes)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {"reference": reference_source, "holdout": holdout_source, "model": model_source},
        "versions": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "evidently"]},
        "settings": {"seed": seed, "sensitivity_seeds": list(range(seed, seed + CONCEPT_SENSITIVITY_SEEDS)),
                     "income_quantile": CONCEPT_INCOME_QUANTILE, "income_cutoff": cutoff,
                     "periods": [{"name": n, "shock_probability": q} for n, q in CONCEPT_PERIODS],
                     "classification_threshold": THRESHOLD, "auc_drop_alert": CONCEPT_AUC_DROP,
                     "recall_drop_alert": CONCEPT_RECALL_DROP},
        "mechanism": "Yq = Y0 OR (MonthlyIncome <= cutoff AND U < q); U ~ Uniform(0,1), compartilhado entre períodos.",
        "limitations": LIMITATIONS,
        "invariants": {"features_identical": True, "probabilities_identical": True, "predicted_classes_identical": True},
        "income_segment_rows": int(segment.sum()), "results": results,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".concept-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "experiment"
        staging.mkdir()
        (staging / "datasets").mkdir()
        (staging / "predictions").mkdir()
        for name, _ in CONCEPT_PERIODS:
            scenarios[name].to_parquet(staging / "datasets" / f"{name}.parquet", index=False)
            predictions[name].to_parquet(staging / "predictions" / f"{name}.parquet", index=False)
            evaluation = performance_report(control, predictions[name])
            evaluation.save_html(str(staging / f"classification_{name}.html"))
            evaluation.save_json(str(staging / f"classification_{name}.json"))
        feature_evaluation = compare_datasets(scenarios[control_name], scenarios[CONCEPT_PERIODS[-1][0]])
        feature_rows = summarize(feature_evaluation, "c3_vs_control")
        if any(row["drift_detected"] for row in feature_rows):
            raise ValueError("As features fixas não deveriam apresentar drift")
        feature_evaluation.save_html(str(staging / "feature_drift.html"))
        feature_evaluation.save_json(str(staging / "feature_drift.json"))
        pd.DataFrame(feature_rows).to_csv(staging / "feature_drift.csv", index=False)
        pd.DataFrame(results).to_csv(staging / "metrics.csv", index=False)
        pd.DataFrame(segments).to_csv(staging / "segment_metrics.csv", index=False)
        sensitivity.to_csv(staging / "sensitivity.csv", index=False)
        audit.to_parquet(staging / "interventions.parquet", index=False)
        write_analysis(staging / "analysis.html", results, segments, sensitivity, cutoff)
        manifest["artifacts"] = [{"file": str(p.relative_to(staging)), "sha256": sha256(p)}
                                 for p in sorted(staging.rglob("*")) if p.is_file()]
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    record(volume=sum(r["rows"] for r in results),
           alert_count=sum(r["performance_alert"] for r in results),
           manifest_path=str(output_dir / "manifest.json"))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/reference/reference.parquet")
    parser.add_argument("--holdout", default="data/production/holdout.parquet")
    parser.add_argument("--model", default="models/current/model.joblib")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default="artifacts/concept_drift")
    args = parser.parse_args(argv)
    manifest = run_experiment(args.reference, args.holdout, args.model, args.seed, args.output_dir)
    for row in manifest["results"]:
        print(f"{row['scenario']}: alterados={row['changed_labels']}, AUC={row['auc']:.6f}, recall={row['recall']:.6f}, alerta={row['performance_alert']}")
    print(f"Análise: {args.output_dir}/analysis.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
