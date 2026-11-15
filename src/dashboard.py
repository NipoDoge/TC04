"""Dashboard de monitoramento em HTML estático e autocontido.

Lê o consolidado do monitoramento, o impacto dos cenários pareados, a análise causal e o log central de eventos e
gera ``reports/dashboard.html``: sem servidor, sem CDN e sem dados de clientes
(apenas agregados). Os gráficos são SVG montados aqui; o script embutido só
acrescenta tooltips, leitura por teclado e o filtro da tabela de alertas, de modo
que todo valor também está visível em rótulos ou nas tabelas de dados.
"""
import argparse
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import sys

from src import config
from src.dashboard_assets import CSS, JS
from src.observability import new_run_id, read_events, setup_logging, track

MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilização do rotativo",
    "DebtRatio": "Comprometimento de renda",
    "MonthlyIncome": "Renda mensal",
    "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos de 30–59 dias",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos de 60–89 dias",
    "NumberOfTimes90DaysLate": "Atrasos de 90 dias ou mais",
    "NumberOfOpenCreditLinesAndLoans": "Linhas de crédito abertas",
    "NumberRealEstateLoansOrLines": "Empréstimos imobiliários",
    "NumberOfDependents": "Dependentes",
    "age": "Idade",
    config.SCORE: "Score do modelo",
}
STAGES = {"ingest": "Ingestão e contrato", "score": "Escoragem", "drift": "Drift", "evaluate": "Avaliação"}
OTHER_STAGES = {"prepare": "Preparação dos dados", "train": "Treino do baseline",
                "simulate": "Simulação dos lotes", "causal": "Análise causal"}
# estado -> (cor de status, ícone, rótulo). A cor nunca aparece sem ícone e rótulo.
STATUS = {
    "success": ("good", "✓", "ok"), "ok": ("good", "✓", "ok"),
    "warning": ("warning", "●", "aviso"), "alert": ("serious", "▲", "alerta"),
    "critical": ("critical", "✕", "crítico"), "rejected": ("critical", "✕", "rejeitado"),
    "failed": ("critical", "✕", "falhou"), "started": ("neutral", "…", "iniciado"),
}
SEVERITIES = {"critical": ("crítico", "críticos"), "alert": ("alerta", "alertas"), "warning": ("aviso", "avisos")}
RULES = {
    "contrato_violado": "Contrato de dados violado", "falha_operacional": "Falha operacional",
    "volume_baixo": "Volume baixo", "drift_feature": "Drift em variável",
    "drift_predicao": "Drift do score", "queda_auc": "Queda de AUC", "queda_recall": "Queda de recall",
    "inadimplencia_elevada": "Inadimplência elevada", "latencia_elevada": "Latência elevada",
}
SCENARIOS = {"controle": "Controle (sem choque)", "data_drift": "Só P(X): data drift",
             "concept_drift": "Só P(Y|X): concept drift", "ambos": "Choque completo"}


# ---------------------------------------------------------------- formatação
def esc(value):
    return html.escape(str(value), quote=True)


def num(value, digits=3):
    """Número no padrão brasileiro: 35.297 e 0,857."""
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(value, digits=1):
    return "—" if value is None else f"{num(value * 100, digits)}%"


def signed(value, digits=3):
    if value is None:
        return "—"
    return ("+" if value >= 0 else "−") + num(abs(value), digits)


def seconds(value):
    return "—" if value is None else f"{num(value, 2)} s"


def month_label(month):
    year, number = str(month).split("-")[:2]
    return f"{MONTHS[int(number) - 1]}/{year[2:]}"


def batch_label(batch):
    """'2026-05' -> 'mai/26'; lotes extras mantêm o sufixo ('entrega corrompida')."""
    parts = str(batch).split("-", 2)
    try:
        label = month_label("-".join(parts[:2]))
    except (ValueError, IndexError):
        return str(batch)
    return label if len(parts) < 3 else f"{label}, {parts[2].replace('-', ' ')}"


def batch_parts(batch):
    """Mês e complemento do lote, para cabeçalhos em duas linhas."""
    label, _, extra = batch_label(batch).partition(", ")
    return label, extra


def feature_label(column):
    return FEATURE_LABELS.get(column, column)


def timestamp(value):
    try:
        return datetime.fromisoformat(value).strftime("%d/%m/%Y %H:%M UTC")
    except (TypeError, ValueError):
        return str(value)


def tip(title, rows, focusable=True):
    """Atributos de um marcador com tooltip (também alcançável pelo teclado).

    Em grades (``focusable=False``) o script distribui o foco: uma parada de Tab
    por tabela e setas entre as células, em vez de uma parada por célula."""
    data = {"t": title, "r": [{"n": name, "v": value} for name, value in rows]}
    label = f"{title}: " + "; ".join(f"{name} {value}" for name, value in rows)
    focus = ' tabindex="0"' if focusable else ""
    return f' data-tip="{esc(json.dumps(data, ensure_ascii=False))}"{focus} aria-label="{esc(label)}"'


def chip(state, label=None, compact=False):
    """Ícone colorido + rótulo. ``compact`` mantém o rótulo só para leitores de tela."""
    color, icon, default = STATUS.get(state, ("neutral", "•", state))
    text = f'<span class="sr">{esc(default)}</span>' if compact else esc(label or default)
    return f'<span class="chip"><span class="ico ico-{color}" aria-hidden="true">{icon}</span>{text}</span>'


def count_label(count, severity):
    return f"{count} {SEVERITIES[severity][count != 1]}"


def table(headers, rows, css="data", right=(), label=None):
    """Tabela de dados; ``right`` lista as colunas numéricas, alinhadas à direita,
    e ``label`` dá um nome à tabela para quem navega por leitor de tela."""
    def align(index):
        return ' class="r"' if index in right else ""

    head = "".join(f'<th scope="col"{align(i)}>{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{align(i)}>{cell}</td>" for i, cell in enumerate(row)) + "</tr>" for row in rows)
    name = f' aria-label="{esc(label)}"' if label else ""
    return f'<div class="scroll"><table class="{css}"{name}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def table_view(headers, rows, label="Dados do gráfico"):
    """Gêmeo tabular de um gráfico: os mesmos valores sem depender de cor ou hover."""
    return ('<details class="table-view"><summary>Ver dados em tabela</summary>'
            f'{table(headers, rows, right=range(1, len(headers)), label=label)}</details>')


# ------------------------------------------------------------------ gráficos
def nice_ticks(low, high, target=5):
    """Marcas redondas que cobrem [low, high]; série constante ganha uma margem."""
    if high - low < 1e-12:
        margin = abs(low) * 0.1 or 1.0
        low, high = low - margin, high + margin
    span = high - low
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw - 1e-12)
    ticks, value = [], math.floor(low / step + 1e-9) * step
    while True:
        ticks.append(round(value, 10))
        if value >= high - 1e-9:
            return ticks
        value += step


def line_chart(labels, series, *, fmt, title, tick_fmt=None, refs=(), notes=(), width=1100, height=230,
               right=70, floor=None, ceiling=None):
    """Linhas no tempo. ``series``: [(nome, valores, cor)]; ``refs``: [(valor, texto, lado)];
    ``notes``: [(índice, valor, texto)] para destacar um ponto."""
    left, top, bottom = 52, 16, 30
    tick_fmt = tick_fmt or fmt
    values = [v for _, data, _ in series for v in data if v is not None] + [r[0] for r in refs]
    low = min(values) if floor is None else floor
    high = max(values) if ceiling is None else ceiling
    ticks = nice_ticks(low, high)
    low, high = ticks[0], ticks[-1]
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / max(len(labels) - 1, 1)

    def x(index):
        return left + step * index

    def y(value):
        return top + plot_h * (1 - (value - low) / (high - low))

    parts = []
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{esc(tick_fmt(tick))}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    for index, label in enumerate(labels):
        parts.append(f'<text class="tick" x="{x(index):.1f}" y="{height - 8}" text-anchor="middle">{esc(label)}</text>')
    for value, text, side in refs:
        anchor, tx = ("end", left + plot_w - 4) if side == "end" else ("start", left + 4)
        parts.append(f'<line class="ref" x1="{left}" x2="{left + plot_w}" y1="{y(value):.1f}" y2="{y(value):.1f}"/>'
                     f'<text class="ref-label" x="{tx}" y="{y(value) - 5:.1f}" text-anchor="{anchor}">{esc(text)}</text>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{top + plot_h}"/>')

    ends = []
    for name, data, color in series:
        points = [(x(i), y(v)) for i, v in enumerate(data) if v is not None]
        if not points:
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f} {py:.1f}" for i, (px, py) in enumerate(points))
        parts.append(f'<path class="line" d="{path}" stroke="{color}"/>')
        parts.extend(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>' for px, py in points)
        last = max(i for i, v in enumerate(data) if v is not None)
        ends.append([y(data[last]), x(last), fmt(data[last])])
    # Rótulos finais: afasta colisões pequenas; se o desvio for grande, o valor
    # fica só na legenda, no tooltip e na tabela (nunca empilhado longe da linha).
    ends.sort()
    for upper, lower in zip(ends, ends[1:]):
        gap = lower[0] - upper[0]
        if gap < 13:
            shift = (13 - gap) / 2
            if shift > 8:
                lower[2] = None
            else:
                upper[0] -= shift
                lower[0] += shift
    parts.extend(f'<text class="end" x="{ex + 9:.1f}" y="{ey + 4:.1f}">{esc(text)}</text>'
                 for ey, ex, text in ends if text is not None)
    for index, value, text in notes:
        previous = next((v for v in reversed(series[0][1][:index]) if v is not None), value)
        anchor, tx = ("start", x(index) + 13) if index == 0 else ("end", x(index) - 13)
        ty = y(value) - 10 if value >= previous else y(value) + 20
        parts.append(f'<circle class="note-ring" cx="{x(index):.1f}" cy="{y(value):.1f}" r="8"/>'
                     f'<text class="note" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}">{esc(text)}</text>')

    hover = [{"x": round(x(i), 1), "t": label,
              "r": [{"n": name, "v": fmt(data[i]), "c": color} for name, data, color in series if data[i] is not None]}
             for i, label in enumerate(labels)]
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" tabindex="0" '
            f'data-points="{esc(json.dumps(hover, ensure_ascii=False))}">{"".join(parts)}</svg>')


def legend(series):
    items = "".join(f'<span class="key"><span class="key-line" style="background:{color}"></span>{esc(name)}</span>'
                    for name, _, color in series)
    return f'<div class="legend">{items}</div>'


def sparkline(values, width=120, height=30):
    """Tendência de um indicador: série em cinza, período atual em destaque."""
    data = [v for v in values if v is not None]
    if len(data) < 2:
        return ""
    low, high = min(data), max(data)
    span = (high - low) or 1.0
    points = [(4 + (width - 8) * i / (len(data) - 1), 4 + (height - 8) * (1 - (v - low) / span))
              for i, v in enumerate(data)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f} {py:.1f}" for i, (px, py) in enumerate(points))
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" aria-hidden="true"><path d="{path}"/>'
            f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="3.5"/></svg>')


def effect_bars(items, *, title, width=540):
    """Barras horizontais a partir do zero. ``items``: [(rótulo, valor, IC ou None)]."""
    row, thick, top, room = 38, 16, 8, 58
    bounds = [0.0] + [v for _, value, ci in items for v in ([value] + list(ci or []))]
    low, high = min(bounds), max(bounds)
    span = (high - low) or 1.0
    left = 178 + (room if low < 0 else 0)          # rótulos + valor à esquerda das barras negativas
    plot_w = width - left - (room if high > 0 else 12)

    def x(value):
        return left + plot_w * (value - low) / span

    height = top + row * len(items) + 22
    zero = x(0.0)
    parts = [f'<line class="axis" x1="{zero:.1f}" x2="{zero:.1f}" y1="{top}" y2="{top + row * len(items)}"/>',
             f'<text class="tick" x="{zero:.1f}" y="{height - 6}" text-anchor="middle">0</text>']
    for index, (label, value, ci) in enumerate(items):
        y0 = top + row * index + (row - thick) / 2
        end, radius = x(value), min(4.0, abs(x(value) - zero))
        inner = end + radius if value < 0 else end - radius
        color = "var(--neg)" if value < 0 else "var(--pos)"
        rows = [("efeito na AUC", signed(value))] + ([("IC 95%", f"{signed(ci[0])} a {signed(ci[1])}")] if ci else [])
        parts.append(
            f'<text class="bar-label" x="170" y="{y0 + thick / 2 + 4:.1f}" text-anchor="end">{esc(label)}</text>'
            f'<path class="bar" fill="{color}" d="M{zero:.1f} {y0:.1f} H{inner:.1f} Q{end:.1f} {y0:.1f} {end:.1f} {y0 + radius:.1f} '
            f'V{y0 + thick - radius:.1f} Q{end:.1f} {y0 + thick:.1f} {inner:.1f} {y0 + thick:.1f} H{zero:.1f} Z"{tip(label, rows)}/>')
        edge = end
        if ci:
            a, b, mid = x(ci[0]), x(ci[1]), y0 + thick / 2
            parts.append(f'<path class="whisker" d="M{a:.1f} {mid:.1f} H{b:.1f} M{a:.1f} {mid - 5:.1f} V{mid + 5:.1f} '
                         f'M{b:.1f} {mid - 5:.1f} V{mid + 5:.1f}"/>')
            edge = a if value < 0 else b
        anchor, tx = ("end", edge - 7) if value < 0 else ("start", edge + 7)
        parts.append(f'<text class="end" x="{tx:.1f}" y="{y0 + thick / 2 + 4:.1f}" text-anchor="{anchor}">{esc(signed(value))}</text>')
    return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def inline_bar(value, scale, negative=False):
    share = 0 if not scale else min(abs(value) / scale, 1) * 100
    return (f'<span class="inline-bar{" neg" if negative else ""}" aria-hidden="true">'
            f'<span style="width:{share:.1f}%"></span></span>')


# -------------------------------------------------------------------- seções
SECTIONS = {"visao": "Visão geral", "tempo": "Linha do tempo", "pipeline": "Saúde do pipeline",
            "drift": "Drift por variável", "cenarios": "Cenários pareados", "modelo": "Desempenho do modelo",
            "alertas": "Alertas", "causal": "Causalidade", "metricas": "Definição das métricas"}
# Grupos de regras do quadro de sinais: um por pergunta que o monitoramento responde.
SIGNALS = (
    ("Contrato de dados", ("contrato_violado",)),
    ("Drift das variáveis", ("drift_feature",)),
    ("Drift do score", ("drift_predicao",)),
    ("Desempenho com rótulo", ("queda_auc", "queda_recall", "inadimplencia_elevada")),
    ("Operação do pipeline", ("falha_operacional", "volume_baixo", "latencia_elevada")),
)
BOARD = {"ok": ("✓", "ok", "good"), "warning": ("●", "aviso", "warning"), "alert": ("▲", "alerta", "serious"),
         "critical": ("✕", "bloqueado", "critical"), "na": ("–", "não avaliado", "neutral")}


def card(title, body, subtitle=None, anchor=""):
    """Seção aberta do boletim: título, subtítulo e conteúdo, separada por um fio."""
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    return f'<section class="block" id="{anchor}"><h2>{esc(title)}</h2>{sub}{body}</section>'


def first_batch(batches, predicate):
    return next((batch for batch in batches if predicate(batch)), None)


def max_feature_psi(batch):
    return max(item["psi"] for column, item in batch["drift"].items() if column != config.SCORE)


def has_alert(batch, rule=None, severity=None):
    return any((rule is None or a["rule"] == rule) and (severity is None or a["severity"] == severity)
               for a in batch["alerts"])


def rail(summary, anchors):
    """Trilho lateral: o que é esta página, de qual execução ela fala e o índice."""
    facts = [("Execução", summary["run_id"]), ("Gerada em", timestamp(summary["generated_at"])),
             ("Modelo publicado", str(summary.get("model_version", "—"))[:12]),
             ("Referência", f'{num(summary["baseline"]["rows"], 0)} linhas de treino')]
    return ('<aside class="rail" aria-label="Execução e índice">'
            '<div><p class="product">Monitoramento do modelo de crédito</p>'
            '<p class="product-sub">TC04, Fase 4</p></div>'
            '<dl class="run">' + "".join(f"<dt>{esc(name)}</dt><dd>{esc(value)}</dd>" for name, value in facts) + "</dl>"
            '<nav class="toc" aria-label="Seções"><ul>'
            + "".join(f'<li><a href="#{anchor}">{esc(SECTIONS[anchor])}</a></li>' for anchor in anchors)
            + '</ul></nav><button type="button" id="theme" class="theme">Usar tema escuro</button></aside>')


def signal_cell(batch, rules):
    """Estado de um grupo de regras em um lote: (estado, linhas do tooltip)."""
    found = [alert for alert in batch["alerts"] if alert["rule"] in rules]
    if found:
        state = next(s for s in ("critical", "alert", "warning") if any(a["severity"] == s for a in found))
        return state, [(alert["message"], RULES.get(alert["rule"], alert["rule"])) for alert in found]
    ingested = batch["stages"].get("ingest", {}).get("status") == "success"
    checked = (batch["status"] == "success" or "falha_operacional" in rules
               or ("contrato_violado" in rules and ingested))
    if not checked:
        return "na", [("o lote foi interrompido antes desta verificação", "não avaliado")]
    return "ok", [("nenhuma regra disparada", "ok")]


def signal_board(summary):
    """Quadro de sinais: lotes nas colunas, grupos de regras nas linhas."""
    batches = summary["batches"]
    head = '<th scope="col"><span class="sr">Sinal</span></th>'
    for batch in batches:
        label, extra = batch_parts(batch["batch"])
        head += f'<th scope="col">{esc(label)}' + (f"<small>{esc(extra)}</small>" if extra else "") + "</th>"
    body = ""
    for title, rules in SIGNALS:
        body += f'<tr><th scope="row">{esc(title)}</th>'
        for batch in batches:
            state, rows = signal_cell(batch, rules)
            icon, label, color = BOARD[state]
            if state == "critical" and any(a["rule"] == "falha_operacional" for a in batch["alerts"] if a["rule"] in rules):
                label = "falhou"
            text = f'<span class="sr">{label}</span>' if state == "ok" else esc(label)
            body += (f'<td class="s-{state}"{tip(f"{title}, {batch_label(batch["batch"])}", rows, focusable=False)}>'
                     f'<span class="ico ico-{color}" aria-hidden="true">{icon}</span>{text}</td>')
        body += "</tr>"
    columns = '<col class="signal">' + "<col>" * len(batches)
    return ('<div class="scroll"><table class="board" data-grid><caption>Sinais por lote, na ordem de chegada. '
            "Passe o cursor em uma célula, ou use Tab e as setas, para ler as mensagens."
            f"</caption><colgroup>{columns}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def section_stats(summary, done):
    """Fatos da execução em uma faixa: volume, latência, alertas e os dois indicadores do último lote."""
    totals, base = summary["totals"], summary["baseline"]
    alerts = totals["alerts"]
    last = done[-1] if done else None
    slowest = max(done, key=lambda b: b["duration_s"]) if done else None

    def stat(label, value, note, extra=""):
        return (f'<div class="stat"><p class="stat-label">{esc(label)}</p><p class="stat-value">{value}</p>'
                f'<p class="stat-note">{note}</p>{extra}</div>')

    rejected = f'{totals["rejected"]} rejeitado{"" if totals["rejected"] == 1 else "s"} pelo contrato de dados'
    stats = [
        stat("Lotes processados", f'{totals["success"]} de {totals["batches"]}',
             f'{rejected}, {totals["failed"]} com falha operacional'),
        stat("Linhas processadas", num(totals["rows_processed"], 0),
             f'{num(totals["rows_blocked"], 0)} bloqueadas na ingestão'),
        stat("Latência mediana por lote", seconds(totals["median_duration_s"]),
             f'máxima {seconds(slowest["duration_s"])} em {esc(batch_label(slowest["batch"]))}' if slowest else "—"),
        stat("Alertas gerados", str(sum(alerts.values())),
             "<br>".join(chip(s, count_label(alerts[s], s)) for s in SEVERITIES)),
    ]
    if last:
        auc, psi_max = last["performance"]["auc"], max_feature_psi(last)
        stats.append(stat(f"AUC em {batch_label(last['batch'])}", num(auc),
                          f'{signed(auc - base["auc"])} contra o baseline ({num(base["auc"])})',
                          sparkline([b["performance"]["auc"] for b in done])))
        stats.append(stat(f"PSI máximo em {batch_label(last['batch'])}", num(psi_max),
                          f'limite de alerta {num(summary["thresholds"]["psi_alert"], 2)}',
                          sparkline([max_feature_psi(b) for b in done])))
    return f'<div class="stats">{"".join(stats)}</div>'


def section_header(summary, done):
    """Abertura: o veredito do último lote e o quadro de sinais de todos eles."""
    last = done[-1] if done else None
    counts = {s: sum(a["severity"] == s for a in (last["alerts"] if last else [])) for s in SEVERITIES}
    worst = next((s for s in ("critical", "alert", "warning") if counts[s]), "ok")
    if last is None:
        verdict = "Nenhum lote foi processado com sucesso nesta execução."
    else:
        found = ", ".join(count_label(count, s) for s, count in counts.items() if count) or "sem alertas"
        verdict = (f"Último lote ({batch_label(last['batch'])}): {found}. AUC {num(last['performance']['auc'])} "
                   f"contra {num(summary['baseline']['auc'])} do baseline; PSI máximo das variáveis "
                   f"{num(max_feature_psi(last))}.")
    return (
        '<header class="hero" id="visao"><h1>Drift, desempenho e saúde do pipeline</h1>'
        f'<p class="verdict v-{STATUS[worst][0]}">{chip(worst, STATUS[worst][2].capitalize())} <span>{esc(verdict)}</span></p>'
        '<p class="synthetic">Os lotes de produção são simulados (choque econômico sintético sobre o holdout). '
        "Os números demonstram o funcionamento do monitoramento, não o desempenho de uma operação real.</p>"
        + signal_board(summary) + section_stats(summary, done) + "</header>")


def section_timeline(summary, done):
    """O achado central: o desempenho cai antes de o drift dos dados disparar."""
    limits, base = summary["thresholds"], summary["baseline"]
    labels = [batch_label(b["batch"]) for b in done]
    auc = [b["performance"]["auc"] for b in done]
    psi_max = [max_feature_psi(b) for b in done]
    auc_alert = first_batch(done, lambda b: has_alert(b, "queda_auc"))
    psi_warning = first_batch(done, lambda b: any(i["status"] != "ok" for c, i in b["drift"].items() if c != config.SCORE))
    psi_alert = first_batch(done, lambda b: any(i["status"] == "alert" for c, i in b["drift"].items() if c != config.SCORE))
    position = {b["batch"]: i for i, b in enumerate(done)}

    auc_notes = [(position[auc_alert["batch"]], auc_alert["performance"]["auc"], "1º alerta de AUC")] if auc_alert else []
    psi_notes = []
    if psi_warning:
        psi_notes.append((position[psi_warning["batch"]], max_feature_psi(psi_warning), "1º aviso de PSI"))
    if psi_alert and psi_alert is not psi_warning:
        psi_notes.append((position[psi_alert["batch"]], max_feature_psi(psi_alert), "1º alerta de PSI"))

    reading = []
    if auc_alert:
        reading.append(
            f'A AUC cruzou o limite de queda ({num(limits["auc_drop"], 2)}) em <strong>{esc(batch_label(auc_alert["batch"]))}</strong>, '
            f'com {num(auc_alert["performance"]["auc"])}; nesse lote o maior PSI entre as variáveis era '
            f'{num(max_feature_psi(auc_alert))}.')
    if psi_warning:
        reading.append(f'O primeiro aviso de PSI (≥ {num(limits["psi_warning"], 2)}) veio em '
                       f'<strong>{esc(batch_label(psi_warning["batch"]))}</strong>'
                       + (f' e o primeiro alerta (≥ {num(limits["psi_alert"], 2)}) em '
                          f'<strong>{esc(batch_label(psi_alert["batch"]))}</strong>.' if psi_alert else "."))
    reading.append("A AUC depende do rótulo maturado (dois anos no caso real): sem ele, o primeiro sinal disponível "
                   "seria o do PSI. É a degradação silenciosa que o monitoramento precisa cobrir.")

    auc_chart = line_chart(
        labels, [("AUC do lote", auc, "var(--series-1)")], fmt=num, tick_fmt=lambda v: num(v, 2),
        title="AUC por lote mensal contra o baseline", notes=auc_notes, height=220,
        refs=[(base["auc"], f'baseline da validação {num(base["auc"])}', "end"),
              (base["auc"] - limits["auc_drop"], f'limite de alerta {num(base["auc"] - limits["auc_drop"])}', "end")])
    psi_chart = line_chart(
        labels, [("PSI máximo", psi_max, "var(--series-1)")], fmt=num, tick_fmt=lambda v: num(v, 2),
        title="PSI máximo entre as variáveis por lote mensal", notes=psi_notes, height=220, floor=0.0,
        refs=[(limits["psi_warning"], f'aviso {num(limits["psi_warning"], 2)}', "start"),
              (limits["psi_alert"], f'alerta {num(limits["psi_alert"], 2)}', "start")])
    rows = [[esc(label), num(a), num(p)] for label, a, p in zip(labels, auc, psi_max)]
    body = (f'<p class="reading">{" ".join(reading)}</p>'
            f'<h3>AUC do lote (rótulos observados)</h3>{auc_chart}'
            f'<h3>PSI máximo entre as variáveis de entrada</h3>{psi_chart}'
            + table_view(["Lote", "AUC", "PSI máximo"], rows))
    return card("Desempenho e drift ao longo do tempo", body,
                "Dois sinais na mesma linha do tempo: o que o modelo acerta e o quanto os dados mudaram.", "tempo")


def section_pipeline(summary, events):
    """Status, duração e volume por etapa e lote, lidos do log central de eventos."""
    by_batch = {}
    for event in events:
        if event.get("run_id") == summary["run_id"] and event.get("stage") in STAGES:
            by_batch.setdefault(event["batch"], {})[event["stage"]] = event
    longest = max((b["duration_s"] or 0 for b in summary["batches"]), default=0) or 1
    rows = []
    for batch in summary["batches"]:
        logged = by_batch.get(batch["batch"], {})
        levels = [batch.get("data_level"), batch.get("concept_level")]
        cells = [f'<span class="strong">{esc(batch_label(batch["batch"]))}</span>',
                 esc(" e ".join(level for level in levels if level)) or "—",
                 chip(batch["status"]), num(batch["rows"], 0) if batch["rows"] is not None else "—"]
        for stage in STAGES:
            info = logged.get(stage) or batch["stages"].get(stage)
            cells.append("—" if info is None else
                         f'{chip(info["status"], compact=True)}<span class="num">{seconds(info.get("duration_s"))}</span>')
        cells.append(f'<span class="num">{seconds(batch["duration_s"])}</span>'
                     f'<span class="meter" aria-hidden="true"><span style="width:{100 * (batch["duration_s"] or 0) / longest:.1f}%"></span></span>')
        counts = [count_label(sum(a["severity"] == s for a in batch["alerts"]), s)
                  for s in SEVERITIES if any(a["severity"] == s for a in batch["alerts"])]
        cells.append(esc(", ".join(counts)) or "—")
        rows.append(cells)
    health = table(["Lote", "Níveis simulados", "Status", "Linhas", *STAGES.values(), "Duração total", "Alertas"], rows,
                   "data health", right=(3,),
                   label="Status, volume e duração por lote e por etapa")

    latest = {}
    for event in events:
        if event.get("stage") in OTHER_STAGES:
            latest[event["stage"]] = event
    other = [[esc(OTHER_STAGES[stage]), chip(e["status"]), esc(timestamp(e["ts"])), seconds(e.get("duration_s")),
              num(e["rows"], 0) if e.get("rows") is not None else "—"]
             for stage, e in ((s, latest.get(s)) for s in OTHER_STAGES) if e]
    run_events = [e for e in events if e.get("run_id") == summary["run_id"]]
    levels = {}
    for event in run_events:
        levels[event.get("level", "INFO")] = levels.get(event.get("level", "INFO"), 0) + 1
    plural = "evento" if len(run_events) == 1 else "eventos"
    log_note = (f'<p class="note">Log central <code>{esc(config.EVENT_LOG)}</code>: {len(run_events)} {plural} nesta execução ('
                + ", ".join(f"{count} {esc(level)}" for level, count in sorted(levels.items())) + ").</p>") if run_events else (
                '<p class="note">Log central indisponível: a tabela usa o consolidado do monitoramento.</p>')
    log_rows = [[esc(e["ts"][11:23]), esc(e["stage"]), esc(e["batch"] or "—"), esc(e["status"]), esc(e["level"]),
                 seconds(e.get("duration_s")), num(e["rows"], 0) if e.get("rows") is not None else "—"]
                for e in run_events]
    log_table = (f'<details class="table-view"><summary>Ver o registro de eventos desta execução ({len(log_rows)})</summary>'
                 + table(["Hora (UTC)", "Etapa", "Lote", "Status", "Nível", "Duração", "Linhas"], log_rows, right=(5, 6),
                         label="Eventos desta execução no log central")
                 + "</details>") if log_rows else ""
    extra = ("<h3>Demais etapas do pipeline (última execução registrada)</h3>"
             + table(["Etapa", "Status", "Quando", "Duração", "Linhas"], other, right=(3, 4),
                     label="Última execução das demais etapas")) if other else ""
    return card("Saúde do pipeline", health + log_note + log_table + extra,
                "Uma linha por lote, na ordem de chegada. Os níveis simulados são os dos experimentos pareados: "
                "data drift (t) e concept drift (c). O primeiro lote inclui a carga inicial do Evidently.", "pipeline")


def psi_level(value, limits):
    warning, alert = limits["psi_warning"], limits["psi_alert"]
    return 1 + sum(value >= round(edge, 10) for edge in (warning / 2, warning, alert, alert * 1.5))


def section_drift(summary, done):
    limits = summary["thresholds"]
    shocked = set(summary.get("simulation", {}).get("shocked_features", []))
    columns = sorted(summary["features"], key=lambda c: -max(b["drift"][c]["psi"] for b in done))
    head = '<th scope="col">Variável</th>'
    for batch in done:
        report = batch.get("report")
        link = ""
        if report and Path(report).exists():
            href = os.path.relpath(report, config.DASHBOARD.parent).replace(os.sep, "/")
            name = f"Relatório do Evidently de {batch_label(batch['batch'])}"
            link = f'<a href="{esc(href)}" aria-label="{esc(name)}" title="{esc(name)}">relatório</a>'
        head += f'<th scope="col">{esc(batch_label(batch["batch"]))}{link}</th>'
    body = ""
    for column in columns + [config.SCORE]:
        tag = '<span class="tag">alterada na simulação</span>' if column in shocked else ""
        body += (f'<tr class="{"score-row" if column == config.SCORE else ""}"><th scope="row">'
                 f'{esc(feature_label(column))}{tag}<code lang="en">{esc(column)}</code></th>')
        for batch in done:
            item = batch["drift"][column]
            mark = {"warning": '<span aria-hidden="true">●</span>', "alert": '<span aria-hidden="true">▲</span>'}.get(item["status"], "")
            verdict = {True: "drift detectado", False: "sem drift"}.get(item.get("evidently_drift"), "fora do relatório")
            rows = [("PSI (decis da referência)", num(item["psi"])), ("situação", STATUS[item["status"]][2]),
                    ("Evidently", verdict),
                    ("KS, estatística D", num(item["ks_stat"])), ("KS, p-valor", f'{item["ks_pvalue"]:.2g}'),
                    ("média, referência → lote", f'{num(item["ref_mean"], 2)} → {num(item["cur_mean"], 2)}'),
                    ("mediana, referência → lote", f'{num(item["ref_median"], 2)} → {num(item["cur_median"], 2)}')]
            body += (f'<td class="seq-{psi_level(item["psi"], limits)}"'
                     f'{tip(f"{feature_label(column)}, {batch_label(batch["batch"])}", rows, focusable=False)}>{mark}{num(item["psi"])}</td>')
        body += "</tr>"
    warning, alert = limits["psi_warning"], limits["psi_alert"]
    scale = [(1, f"abaixo de {num(warning / 2, 2)}"), (2, f"{num(warning / 2, 2)} a {num(warning, 2)}"),
             (3, f"● aviso: {num(warning, 2)} a {num(alert, 2)}"), (4, f"▲ alerta: {num(alert, 2)} a {num(alert * 1.5, 2)}"),
             (5, f"▲ alerta: {num(alert * 1.5, 2)} ou mais")]
    legend_html = '<div class="legend scale">' + "".join(
        f'<span class="key"><span class="swatch seq-{level}"></span>{esc(text)}</span>' for level, text in scale) + "</div>"
    note = ('<p class="note">As cores mostram o tamanho da mudança: PSI com faixas fixas nos decis da referência (treino), '
            "a mesma medida gravada no resumo dos cenários pareados. As regras de alerta usam esse tamanho. Os links “relatório” "
            "abrem o relatório do Evidently de cada lote, que diz se há drift por feature (KS nas medidas contínuas, PSI de "
            "categorias nas demais); o KS é mais sensível e acusa também mudanças pequenas. Passe o cursor em uma célula, ou use "
            "Tab e as setas, para ver o veredito do Evidently, o KS, as médias e as medianas. Os relatórios são gerados por "
            "<code>make report</code>; o repositório versiona apenas alguns deles (ver README).</p>")
    return card("Drift por variável e por lote",
                '<div class="scroll"><table class="heat" data-grid><caption class="sr">PSI por variável (linhas) e por lote (colunas)'
                f'</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
                + legend_html + note,
                "PSI de cada variável de entrada e do score, lote contra referência. Variáveis ordenadas pelo maior PSI.", "drift")


def section_scenarios(scenarios, limits):
    """Cenários pareados da Etapa 2 (src.simulate_drift): mesmos clientes e rótulos, só X muda."""
    if not scenarios:
        return ""
    shifted = [c for c in scenarios.get("shifted_columns", config.DRIFT_SHIFTED_COLUMNS)
               if all(c in item["drift"] for item in scenarios["scenarios"])]

    def psi_cell(item):
        mark = {"warning": "● ", "alert": "▲ "}.get(item["status"], "")
        return f'<span class="pill seq-{psi_level(item["psi"], limits)}">{mark}{num(item["psi"])}</span>'

    rows = []
    for item in scenarios["scenarios"]:
        metrics, drift, factors = item["performance"], item["drift"], item.get("factors") or {}
        link = ""
        if Path(item["report"]).exists():
            href = os.path.relpath(item["report"], config.DASHBOARD.parent).replace(os.sep, "/")
            name = f"Relatório do Evidently do cenário {item['scenario']}"
            link = f'<a href="{esc(href)}" aria-label="{esc(name)}" title="{esc(name)}">relatório</a>'
        applied = "; ".join(f"{feature_label(column).lower()} ×{num(value, 2)}" for column, value in factors.items())
        flagged = item.get("evidently_drifted")
        verdict = "—" if flagged is None else (f'{len(flagged)} de {len(drift) - 1}' if flagged else "nenhuma")
        rows.append([f'<span class="strong">{esc(item["scenario"])}</span>'
                     + (f'<span class="factors">{esc(applied)}</span>' if applied else ""),
                     verdict, *[psi_cell(drift[column]) for column in shifted], psi_cell(drift[config.SCORE]),
                     num(metrics["auc"]), pct(metrics["flagged_rate"]), link or "—"])
    first, last = scenarios["scenarios"][0], scenarios["scenarios"][-1]
    feature, largest = max(((c, last["drift"][c]) for c in shifted), key=lambda pair: pair[1]["psi"])
    reading = (
        f'No cenário mais forte ({esc(last["scenario"])}), a maior mudança é em {esc(feature_label(feature).lower())} '
        f'(PSI {num(largest["psi"])}) e o score do modelo tem PSI de {num(last["drift"][config.SCORE]["psi"])}. '
        f'A parcela sinalizada vai de {pct(first["performance"]["flagged_rate"])} para {pct(last["performance"]["flagged_rate"])}, '
        f'enquanto a AUC vai de {num(first["performance"]["auc"])} para {num(last["performance"]["auc"])}: '
        "com os rótulos preservados, o data drift muda as decisões do modelo mais do que a sua capacidade de ordenar.")
    headers = ["Cenário", "Colunas em drift no Evidently", *[f"PSI: {feature_label(c).lower()}" for c in shifted],
               "PSI do score", "AUC", "Sinalizados", "Relatório"]
    body = (f'<p class="reading">{reading}</p>'
            + table(headers, rows, right=(len(shifted) + 3, len(shifted) + 4), label="Drift e impacto por cenário pareado")
            + f'<p class="note">{esc(scenarios.get("note", ""))} Os {num(first["rows"], 0)} clientes são os mesmos em todos os '
            "cenários; os fatores multiplicam as colunas do holdout original, e quem passa a usar todo o limite rotativo ganha "
            "um atraso de 30–59 dias. A coluna do Evidently vem de <code>src.detect_drift</code> (KS nas medidas contínuas e "
            "PSI de categorias nas demais), que diz se há drift; o PSI por decis mede o tamanho da mudança.</p>")
    return card("Cenários pareados da Etapa 2", body,
                "Controle e três intervenções sobre o holdout inteiro: quanto os dados mudaram e quanto o modelo sentiu.",
                "cenarios")


def section_model(summary, done):
    limits, base = summary["thresholds"], summary["baseline"]
    labels = [batch_label(b["batch"]) for b in done]

    def metric(name):
        return [b["performance"][name] for b in done]

    recall = line_chart(labels, [("Recall", metric("recall"), "var(--series-1)")], fmt=num, tick_fmt=lambda v: num(v, 2),
                        title="Recall por lote", width=540, height=210, right=58,
                        refs=[(base["recall"], f'baseline {num(base["recall"])}', "end"),
                              (base["recall"] - limits["recall_drop"], f'limite {num(base["recall"] - limits["recall_drop"])}', "end")])
    ks = line_chart(labels, [("KS do modelo", metric("ks"), "var(--series-1)")], fmt=num, tick_fmt=lambda v: num(v, 2),
                    title="KS do modelo por lote", width=540, height=210, right=58,
                    refs=[(base["ks"], f'baseline {num(base["ks"])}', "end")])
    rates = [("Inadimplência observada", metric("default_rate"), "var(--series-1)"),
             ("Score médio", metric("mean_score"), "var(--series-2)"),
             ("Sinalizados (score ≥ " + num(limits["decision_threshold"], 1) + ")", metric("flagged_rate"), "var(--series-3)")]
    default_limit = base["default_rate"] * limits["default_rate_factor"]
    rates_chart = line_chart(labels, rates, fmt=pct, tick_fmt=lambda v: pct(v, 0),
                             title="Inadimplência observada, score médio e parcela sinalizada por lote", height=250, floor=0.0,
                             refs=[(default_limit, f'alerta de inadimplência {pct(default_limit)} '
                                    f'({num(limits["default_rate_factor"], 1)}× a referência)', "start")])
    rows = [[esc(label), num(b["performance"]["auc"]), num(b["performance"]["recall"]), num(b["performance"]["precision"]),
             num(b["performance"]["ks"]), pct(b["performance"]["default_rate"]), pct(b["performance"]["mean_score"]),
             pct(b["performance"]["flagged_rate"])] for label, b in zip(labels, done)]
    rows.insert(0, ["<em>referência / baseline</em>", num(base["auc"]), num(base["recall"]), "—", num(base["ks"]),
                    pct(base["default_rate"]), pct(base["mean_score"]), pct(base["flagged_rate"])])
    body = (f'<div class="grid2"><div><h3>Recall no limiar {num(limits["decision_threshold"], 1)}</h3>{recall}</div>'
            f'<div><h3>KS do modelo (separação entre bons e maus)</h3>{ks}</div></div>'
            f'<h3>O que o modelo prevê e o que acontece</h3>{legend(rates)}{rates_chart}'
            '<p class="note">O score médio não é uma probabilidade calibrada: o treino pondera a classe positiva. '
            "Vale a comparação com a própria referência, não o nível absoluto.</p>"
            + table_view(["Lote", "AUC", "Recall", "Precisão", "KS", "Inadimplência", "Score médio", "Sinalizados"], rows))
    return card("Desempenho do modelo", body,
                "Métricas com rótulo observado, comparadas ao baseline medido na validação (fora do treino).", "modelo")


def section_alerts(summary):
    alerts = [alert for batch in summary["batches"] for alert in batch["alerts"]]
    counts = summary["totals"]["alerts"]
    buttons = f'<button type="button" data-filter="all" aria-pressed="true">Todos ({len(alerts)})</button>' + "".join(
        f'<button type="button" data-filter="{s}" aria-pressed="false">{STATUS[s][2].capitalize()} ({counts[s]})</button>'
        for s in SEVERITIES)
    rows = "".join(
        f'<tr data-sev="{esc(a["severity"])}"><td>{chip(a["severity"])}</td><td>{esc(batch_label(a["batch"]))}</td>'
        f'<td>{esc(RULES.get(a["rule"], a["rule"]))}</td><td>{esc(a["message"])}</td></tr>' for a in alerts)
    head = "".join(f'<th scope="col">{h}</th>' for h in ["Severidade", "Lote", "Regra", "Mensagem"])
    shown = f"{len(alerts)} alerta exibido" if len(alerts) == 1 else f"{len(alerts)} alertas exibidos"
    body = (f'<div class="filters" role="group" aria-label="Filtrar por severidade">{buttons}'
            f'<output id="alert-count" aria-live="polite">{shown}</output></div>'
            '<div class="scroll"><table class="data alerts" aria-label="Alertas gerados nesta execução">'
            f'<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
            if alerts else '<p class="note">Nenhum alerta nesta execução.</p>')
    return card("Alertas", body, "Crítico interrompe o lote; alerta exige investigação; aviso pede acompanhamento. "
                "Um alerta de drift indica mudança de distribuição, não a sua causa.", "alertas")


def section_causal(causal):
    if not causal:
        return card("Atribuição causal", '<p class="note">Análise causal ainda não executada '
                    "(<code>python -m src.causal_analysis</code>).</p>", anchor="causal")
    scenarios, parts = causal["scenarios"], causal["auc_decomposition"]
    rows = []
    for name, item in scenarios.items():
        ci, m = item["auc_effect_ci95"], item["metrics"]
        rows.append([f'<span class="strong">{esc(SCENARIOS.get(name, name))}</span>', num(m["auc"]),
                     "—" if ci is None else f'{signed(item["effect"]["auc"])} <span class="muted">[{signed(ci[0])}; {signed(ci[1])}]</span>',
                     num(m["recall"]), pct(m["flagged_rate"]), pct(m["default_rate"]), num(item["psi_score"])])
    bars = effect_bars(
        [("Só P(X): data drift", parts["data_drift"], scenarios["data_drift"]["auc_effect_ci95"]),
         ("Só P(Y|X): concept drift", parts["concept_drift"], scenarios["concept_drift"]["auc_effect_ci95"]),
         ("Interação entre os dois", parts["interaction"], None),
         ("Choque completo (total)", parts["total"], scenarios["ambos"]["auc_effect_ci95"])],
        title="Decomposição do efeito na AUC por mecanismo")
    interventions = causal["feature_interventions"]
    scale_auc = max((abs(i["effect"]["auc"]) for i in interventions.values()), default=0)
    scale_score = max((abs(i["effect"]["mean_score"]) for i in interventions.values()), default=0)
    feature_rows = [
        [esc(feature_label(column)),
         f'<span class="num">{signed(item["effect"]["auc"])}</span>{inline_bar(item["effect"]["auc"], scale_auc, True)}',
         f'<span class="num">{signed(item["effect"]["mean_score"])}</span>{inline_bar(item["effect"]["mean_score"], scale_score)}',
         signed(item["effect"]["flagged_rate"] * 100, 1) + " p.p."]
        for column, item in sorted(interventions.items(), key=lambda pair: pair[1]["effect"]["auc"])]
    body = (
        f'<p class="reading">Mesmo modelo, mesmos {num(causal["rows"], 0)} clientes de {esc(batch_label(causal["month"]))} e mesmo sorteio: '
        f'cada cenário liga uma parte do choque daquele lote por vez (níveis {esc(causal.get("data_level", "—"))} e '
        f'{esc(causal.get("concept_level", "—"))}). O que muda entre cenários é efeito da intervenção, dentro da simulação.</p>'
        '<div class="grid2"><div><h3>Efeito na AUC por mecanismo (IC de 95%, bootstrap pareado)</h3>'
        f'{bars}<p class="note">Barras a partir do zero; o traço fino é o intervalo de confiança. '
        "A interação é o que sobra: os dois mecanismos juntos derrubam menos do que a soma dos efeitos isolados.</p></div>"
        "<div><h3>Intervenção em uma variável por vez (rótulos originais)</h3>"
        + table(["Variável", "Efeito na AUC", "Efeito no score médio", "Sinalizados"], feature_rows, "data causal", right=(3,),
                label="Efeito de alterar uma variável por vez")
        + "</div></div><h3>Cenários contrafactuais</h3>"
        + table(["Cenário", "AUC", "Efeito na AUC [IC 95%]", "Recall", "Sinalizados", "Inadimplência", "PSI do score"], rows,
                right=(1, 3, 4, 5, 6), label="Métricas por cenário contrafactual")
        + '<p class="note">Limite da análise: as intervenções são sintéticas e conhecidas. O resultado mostra como o diagnóstico '
        "seria conduzido e quanto cada mecanismo pesa neste cenário; não identifica a causa de uma degradação real.</p>")
    return card("Atribuição causal por intervenção", body,
                "O que derruba o modelo: a mudança nos dados, P(X), ou a mudança na relação com a inadimplência, P(Y|X)?", "causal")


def section_definitions(summary):
    t = summary["thresholds"]
    rows = [
        ["Status das etapas", "Desfecho de cada etapa por lote: <code>success</code>, <code>rejected</code> (contrato) ou <code>failed</code> (falha operacional).",
         "Evento por etapa no log central.", "Rejeição ou falha gera alerta crítico e interrompe o lote."],
        ["Latência", "Tempo de parede de cada etapa e do lote (soma das etapas), em segundos.",
         "<code>duration_s</code> de cada evento.", f"Aviso se o lote levar mais de {num(t['latency_factor'], 0)}× a mediana dos anteriores (mínimo de 3)."],
        ["Volume", "Linhas recebidas, processadas e bloqueadas por lote.", "<code>rows</code> de cada evento.",
         f"Aviso se o lote tiver menos de {num(t['min_rows'], 0)} linhas."],
        ["Alertas", "Regras disparadas por lote, por severidade.", "Evento <code>alert</code> por regra disparada.",
         "Crítico, alerta ou aviso, conforme a regra."],
        ["PSI", "Population Stability Index da variável no lote contra a referência, com faixas nos decis da referência.",
         "Evidently (<code>psi_decis</code>) por variável e para o score.", f"Aviso ≥ {num(t['psi_warning'], 2)}; alerta ≥ {num(t['psi_alert'], 2)}."],
        ["KS de duas amostras", "Maior distância entre as distribuições acumuladas (D) e p-valor.", "Evidently, por variável.",
         "Sem alerta: evidência complementar."],
        ["AUC e recall", "Ordenação e cobertura dos inadimplentes no lote, com rótulo observado.", f"Recall no limiar {num(t['decision_threshold'], 1)}.",
         f"Alerta se a AUC cair ≥ {num(t['auc_drop'], 2)} ou o recall ≥ {num(t['recall_drop'], 2)} contra o baseline."],
        ["Inadimplência observada", "Proporção de inadimplentes no lote.", "Média do alvo no lote.",
         f"Alerta se ≥ {num(t['default_rate_factor'], 1)}× a taxa da referência."],
        ["Sinalizados", "Parcela do lote com score ≥ limiar de decisão.", "Calculada na escoragem.",
         "Sem regra própria; lida junto com o PSI do score."],
    ]
    return card("Definição das métricas", table(["Métrica", "O que mede", "Como é obtida", "Regra de alerta"], rows, "data defs",
                                                label="Definição das métricas e regras de alerta"),
                "Os limiares são pontos de partida para a simulação e ficam em <code>src/config.py</code>.", "metricas")


def footer(summary):
    simulation = summary.get("simulation", {})
    schedule = "; ".join(f'{month_label(item["month"])} {item["data_level"]} e {item["concept_level"]}'
                         for item in simulation.get("schedule", []))
    return (f'<footer><p>Fontes: <code>{esc(config.SUMMARY)}</code>, <code>{esc(config.SCENARIO_IMPACT)}</code>, '
            f'<code>{esc(config.CAUSAL)}</code> e <code>{esc(config.EVENT_LOG)}</code>. '
            f'Lotes simulados com seed {esc(simulation.get("seed", "—"))} sobre <code>{esc(simulation.get("source", "—"))}</code>; '
            f'níveis de data drift e de concept drift por mês: {esc(schedule) or "—"}.</p>'
            "<p>O dashboard contém somente agregados por lote; nenhuma linha de cliente é gravada aqui ou no log central.</p></footer>")


def build(summary, causal=None, events=(), scenarios=None):
    """Monta o HTML completo a partir dos consolidados (sem tocar em disco)."""
    done = [batch for batch in summary["batches"] if batch["status"] == "success"]
    sections = [section_header(summary, done)]
    if done:
        sections += [section_timeline(summary, done), section_pipeline(summary, list(events)),
                     section_drift(summary, done), section_scenarios(scenarios, summary["thresholds"]),
                     section_model(summary, done)]
    else:
        sections += [section_pipeline(summary, list(events)), section_scenarios(scenarios, summary["thresholds"])]
    sections += [section_alerts(summary), section_causal(causal), section_definitions(summary)]
    content = "".join(sections)
    anchors = [anchor for anchor in SECTIONS if f' id="{anchor}"' in content]     # índice só do que existe
    return ('<!DOCTYPE html>\n<html lang="pt-BR"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Monitoramento do modelo de crédito, TC04</title>"
            f'<style>{CSS}</style></head><body class="viz-root">'
            '<a class="skip" href="#conteudo">Pular para o conteúdo</a>'
            f'<div class="layout">{rail(summary, anchors)}<main class="main" id="conteudo">{content}{footer(summary)}</main></div>'
            f'<div id="tip" aria-hidden="true" hidden></div><script>{JS}</script></body></html>\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gera o dashboard HTML do monitoramento.")
    parser.add_argument("--output", default=str(config.DASHBOARD), help="Arquivo HTML de destino")
    args = parser.parse_args(argv)
    if not config.SUMMARY.exists():
        print(f"Resumo não encontrado: {config.SUMMARY}. Execute antes `python -m src.monitor`.", file=sys.stderr)
        return 2
    with track("dashboard") as info:
        summary = json.loads(config.SUMMARY.read_text(encoding="utf-8"))
        causal = json.loads(config.CAUSAL.read_text(encoding="utf-8")) if config.CAUSAL.exists() else None
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        scenarios = (json.loads(config.SCENARIO_IMPACT.read_text(encoding="utf-8"))
                     if config.SCENARIO_IMPACT.exists() else None)
        output.write_text(build(summary, causal, read_events(), scenarios), encoding="utf-8")
        info.update(rows=len(summary["batches"]), output=str(output), bytes=output.stat().st_size)
    print(f"Dashboard: {output} ({output.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    setup_logging()
    new_run_id("dash")
    sys.exit(main())
