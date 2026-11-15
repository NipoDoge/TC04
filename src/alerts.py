"""Regras de alerta do monitoramento. Os limiares estão em ``src.config``.

Severidades: ``critical`` interrompe o lote; ``alert`` exige investigação;
``warning`` pede acompanhamento. Um alerta de drift indica mudança de
distribuição, não a sua causa.
"""
import re
from statistics import median

from src import config

SEVERITIES = ("critical", "alert", "warning")
CHECK_LABELS = {
    "not_nullable": "valor nulo",
    "linhas duplicadas": "linhas duplicadas",
    "column_in_dataframe": "coluna ausente",
    "column_in_schema": "coluna inesperada",
    "finite": "valor infinito",
    "non_empty": "lote vazio",
}


CHECK_PATTERNS = (
    (r"in_range\((.+), (.+)\)", r"fora do intervalo de \1 a \2"),
    (r"greater_than_or_equal_to\((.+)\)", r"menor que \1"),
    (r"isin\((.+)\)", r"fora dos valores \1"),
    (r"dtype\('(.+)'\)", r"tipo diferente de \1"),
)


def br(value, digits=3):
    """Decimal com vírgula, como no restante dos relatórios."""
    return f"{value:.{digits}f}".replace(".", ",")


def describe_check(check):
    for pattern, text in CHECK_PATTERNS:
        if re.fullmatch(pattern, check):
            return re.sub(pattern, text, check)
    return CHECK_LABELS.get(check, check)


def describe_violation(item):
    label = describe_check(item["check"])
    target = f"{item['column']}: " if item["column"] else ""
    return f"{target}{label} ({item['rows']} linhas)"


def evaluate(batch, *, ingest_code=0, violations=(), error=None, rows=None, drift=None,
             performance=None, baseline=None, duration_s=None, previous_durations=()):
    """Aplica as regras a um lote e devolve a lista de alertas gerados."""
    alerts = []

    def add(severity, rule, message, **data):
        alerts.append({"batch": batch, "severity": severity, "rule": rule, "message": message, **data})

    if ingest_code == 1:
        detail = "; ".join(describe_violation(item) for item in violations)
        add("critical", "contrato_violado", f"Lote bloqueado pelo contrato de dados: {detail}")
        return alerts
    if ingest_code == 2:
        add("critical", "falha_operacional", f"Falha operacional na ingestão: {error}")
        return alerts

    if rows is not None and rows < config.MIN_ROWS:
        add("warning", "volume_baixo",
            f"Lote com {rows} linhas, abaixo do mínimo de {config.MIN_ROWS} para a janela de análise",
            value=rows, threshold=config.MIN_ROWS)

    for column, item in (drift or {}).items():
        if item["status"] == "ok":
            continue
        is_score = column == config.SCORE
        subject = "do score do modelo" if is_score else f"em {column}"
        limit = config.PSI_ALERT if item["status"] == "alert" else config.PSI_WARNING
        add(item["status"], "drift_predicao" if is_score else "drift_feature",
            f"Drift {subject}: PSI {br(item['psi'])} (limite {br(limit, 2)})",
            column=column, value=item["psi"], threshold=limit)

    if performance and baseline:
        if performance.get("auc") is not None and baseline["auc"] - performance["auc"] >= config.AUC_DROP:
            add("alert", "queda_auc",
                f"AUC caiu de {br(baseline['auc'])} para {br(performance['auc'])} "
                f"(queda de {br(baseline['auc'] - performance['auc'])}, limite {br(config.AUC_DROP, 2)})",
                value=performance["auc"], threshold=baseline["auc"] - config.AUC_DROP)
        if baseline["recall"] - performance["recall"] >= config.RECALL_DROP:
            add("alert", "queda_recall",
                f"Recall caiu de {br(baseline['recall'])} para {br(performance['recall'])} "
                f"(limite de queda {br(config.RECALL_DROP, 2)})",
                value=performance["recall"], threshold=baseline["recall"] - config.RECALL_DROP)
        limit = baseline["default_rate"] * config.DEFAULT_RATE_FACTOR
        if performance["default_rate"] >= limit:
            add("alert", "inadimplencia_elevada",
                f"Inadimplência observada de {br(performance['default_rate'] * 100, 1)}%, "
                f"{br(performance['default_rate'] / baseline['default_rate'], 1)}x a da referência "
                f"({br(baseline['default_rate'] * 100, 1)}%)",
                value=performance["default_rate"], threshold=limit)

    if duration_s is not None and len(previous_durations) >= 3:
        limit = config.LATENCY_FACTOR * median(previous_durations)
        if duration_s > limit:
            add("warning", "latencia_elevada",
                f"Processamento levou {br(duration_s, 1)} s, acima de {config.LATENCY_FACTOR:.0f}x "
                f"a mediana das execuções anteriores (limite de {br(limit, 1)} s)",
                value=duration_s, threshold=limit)
    return alerts


def count_by_severity(alerts):
    return {severity: sum(1 for alert in alerts if alert["severity"] == severity)
            for severity in SEVERITIES}
