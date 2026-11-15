"""Gera o vídeo STAR do TC04: quadros (HTML -> PNG), narração (voz do macOS) e MP4.

Uso: python docs/video/gera_video.py <raiz do projeto> <pasta de trabalho> [quadros|audio|video|roteiro|tudo]

Requer macOS (comando `say`, voz pt-BR Luciana), Google Chrome, Pillow, NumPy e um ffmpeg
(variável FFMPEG, `ffmpeg` no PATH ou o pacote imageio-ffmpeg). Não faz parte do pipeline
nem das dependências do projeto. Cada frase do roteiro vira um quadro com a legenda e um
trecho de áudio; a duração do quadro é a da fala. Nada aqui lê dados de clientes: os quadros
usam as capturas de docs/img e números agregados dos relatórios.
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
MODE = sys.argv[3] if len(sys.argv) > 3 else "tudo"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VOICE, RATE, SR = "Luciana", 175, 48000
GAP, GAP_SCENE, GAP_STAGE, LEAD_IN, TAIL = 0.35, 0.65, 0.95, 0.5, 1.6
# Grafias só para a fala, escolhidas conferindo a narração com um reconhecedor de voz:
# com "drift" a palavra saía pouco clara em frases corridas.
PRONUNCIA = [(r"\bdrift\b", "drífit")]


def ffmpeg_exe():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ------------------------------------------------------------------ roteiro
# (estágio STAR, cena, passo, legenda, fala). A fala só difere da legenda na grafia de
# siglas, números e termos em inglês, para a voz sintetizada pronunciar direito.
SCRIPT = [
    ("", "titulo", 1,
     "Este é o nosso trabalho da fase 4: drift, observabilidade e governança de um modelo de crédito.",
     "Este é o nosso trabalho da fase quatro: drift, observabilidade e governança de um modelo de crédito."),

    ("S", "situacao", 1,
     "Uma fintech usa um modelo de pontuação de crédito para estimar o risco de inadimplência de cada cliente.",
     "Uma fintéqui usa um modelo de pontuação de crédito para estimar o risco de inadimplência de cada cliente."),
    ("S", "situacao", 2,
     "O risco é a degradação silenciosa: a economia e o perfil dos clientes mudam, e o modelo piora sem que ninguém perceba.",
     None),
    ("S", "situacao", 3,
     "Trabalhamos com uma base pública de crédito, com 150 mil registros, e um modelo XGBoost como referência.",
     "Trabalhamos com uma base pública de crédito, com cento e cinquenta mil registros, e um modelo xis gê búst como referência."),
    ("S", "situacao", 4,
     "Na validação, ele tem AUC de 0,857 e identifica 77% dos inadimplentes.",
     "Na validação, ele tem á u cê de zero vírgula oitocentos e cinquenta e sete, e identifica setenta e sete por cento dos inadimplentes."),

    ("T", "tarefa", 0, "Nossa tarefa tinha quatro partes.", None),
    ("T", "tarefa", 1, "Primeiro: validar os dados que chegam, com um contrato de dados.", None),
    ("T", "tarefa", 2,
     "Segundo: simular e detectar drift, separando a mudança nos dados da mudança na relação com a inadimplência.", None),
    ("T", "tarefa", 3, "Terceiro: observar o processamento, com registros, métricas e alertas centralizados.", None),
    ("T", "tarefa", 4,
     "E quarto: documentar a governança, com proteção de dados, vieses, causalidade e resposta à degradação.", None),

    ("A", "contrato", 1,
     "Começamos pelo contrato de dados, escrito em Pandera: colunas e tipos definidos, faixas válidas, sem valores nulos e sem linhas duplicadas.",
     None),
    ("A", "bloqueio", 1,
     "Na simulação, uma entrega corrompida chega em maio. O contrato bloqueia as 4.612 linhas e manda o lote para a quarentena.",
     "Na simulação, uma entrega corrompida chega em maio. O contrato bloqueia as quatro mil seiscentas e doze linhas e manda o lote para a quarentena."),
    ("A", "bloqueio", 2,
     "Os lotes com drift passam pelo contrato: são dados válidos, que apenas mudaram de distribuição.", None),
    ("A", "niveis", 1,
     "Para o drift, definimos os níveis de intensidade uma única vez e usamos os mesmos em todos os experimentos.", None),
    ("A", "niveis", 2,
     "No data drift, a renda cai, o endividamento e o uso do rotativo sobem, e quem passa a usar todo o limite ganha um atraso.",
     "No dêita drift, a renda cai, o endividamento e o uso do rotativo sobem, e quem passa a usar todo o limite ganha um atraso."),
    ("A", "niveis", 3,
     "No concept drift, bons pagadores de baixa renda passam a inadimplir, sem nenhuma mudança nas variáveis.",
     "No cóncept drift, bons pagadores de baixa renda passam a inadimplir, sem nenhuma mudança nas variáveis."),
    ("A", "niveis", 4,
     "Aplicamos cada nível à base inteira, um mecanismo por vez, e depois a oito lotes mensais, que combinam os dois.", None),
    ("A", "evidently", 1,
     "A detecção usa o Evidently: o teste KS, para as medidas contínuas, e o PSI, para as contagens.",
     "A detecção usa o Évidentli: o teste K S, para as medidas contínuas, e o P S I, para as contagens."),
    ("A", "efeito", 1,
     "Com muitos dados, o teste acusa até mudanças mínimas. Por isso gravamos também o tamanho do efeito.", None),
    ("A", "efeito", 2,
     "No cenário mais fraco, o teste já aponta drift, mas o PSI da renda é de apenas 0,03. No mais forte, chega a 0,60.",
     "No cenário mais fraco, o teste já aponta drift, mas o P S I da renda é de apenas zero vírgula zero três. No mais forte, chega a zero vírgula sessenta."),
    ("A", "pipeline", 1,
     "No monitoramento, cada lote passa por contrato, escoragem, drift, avaliação e regras de alerta.", None),
    ("A", "logmlflow", 1,
     "Cada etapa grava um evento em um log central, e as métricas de cada lote vão para o MLflow.",
     "Cada etapa grava um evento em um log central, e as métricas de cada lote vão para o eme éle flôu."),
    ("A", "painel", 1,
     "Tudo aparece em um painel único, em um só arquivo, que abre direto no navegador.", None),
    ("A", "governanca", 1,
     "Na governança, propomos a base legal de proteção do crédito, a minimização dos dados e prazos de retenção.", None),
    ("A", "governanca", 2,
     "Relatórios e logs guardam apenas valores agregados, e documentamos riscos de viés e um plano de resposta.", None),

    ("R", "tempo", 1,
     "Em oito meses simulados, a AUC caiu de 0,85 para 0,62.",
     "Em oito meses simulados, a á u cê caiu de zero vírgula oitenta e cinco para zero vírgula sessenta e dois."),
    ("R", "tempo", 2,
     "O ponto mais importante é maio: a AUC cai para 0,77 e nenhum PSI das entradas passa do limite de aviso.",
     "O ponto mais importante é maio: a á u cê cai para zero vírgula setenta e sete, e nenhum P S I das entradas passa do limite de aviso."),
    ("R", "tempo", 3,
     "É o efeito do concept drift. Só o rótulo, que no caso real leva até dois anos para maturar, revela essa queda.",
     "É o efeito do cóncept drift. Só o rótulo, que no caso real leva até dois anos para maturar, revela essa queda."),
    ("R", "decisoes", 1,
     "O data drift se comporta de outro jeito: quase não muda a ordenação dos clientes, mas muda as decisões.",
     "O dêita drift se comporta de outro jeito: quase não muda a ordenação dos clientes, mas muda as decisões."),
    ("R", "decisoes", 2,
     "No nível mais forte, o modelo passa a sinalizar cerca de 38% dos clientes, contra 25%, com a mesma inadimplência.",
     "No nível mais forte, o modelo passa a sinalizar cerca de trinta e oito por cento dos clientes, contra vinte e cinco por cento, com a mesma inadimplência."),
    ("R", "causal", 1,
     "O experimento contrafactual confirma: o concept drift tira 0,18 da AUC; o data drift, só 0,01.",
     "O experimento contrafactual confirma: o cóncept drift tira zero vírgula dezoito da á u cê; o dêita drift, só zero vírgula zero um."),
    ("R", "alertas", 1,
     "O monitoramento gerou 21 alertas, do bloqueio pelo contrato às quedas de desempenho.",
     "O monitoramento gerou vinte e um alertas, do bloqueio pelo contrato às quedas de desempenho."),
    ("R", "limites", 1,
     "Tudo isso é simulação: mostra que o monitoramento funciona, não o desempenho de uma operação real.", None),
    ("R", "conclusao", 1,
     "A conclusão é que vigiar as entradas não basta. É preciso acompanhar os rótulos, as decisões e o desempenho por segmento.",
     None),
    ("R", "conclusao", 2,
     "O código, os relatórios e a documentação estão no repositório. Obrigado.", None),
]

# ------------------------------------------------------------------ recortes das capturas
IMG = ROOT / "docs" / "img"
CROPS = {   # nome: (arquivo, caixa em pixels da captura)
    "sinais": (IMG / "dashboard-visao-geral.png", (90, 540, 2440, 1165)),
    "painel": (IMG / "dashboard-visao-geral.png", (60, 0, 2470, 1570)),
    "auc": (IMG / "dashboard-tempo.png", (90, 470, 2440, 1010)),
    "auc_psi": (IMG / "dashboard-tempo.png", (90, 470, 2440, 1590)),
    "cenarios": (IMG / "dashboard-cenarios.png", (90, 385, 2440, 1055)),
    "causal": (IMG / "dashboard-causal.png", (90, 425, 2440, 990)),
    "pipeline": (IMG / "dashboard-pipeline.png", (90, 205, 2440, 965)),
    "alertas": (IMG / "dashboard-alertas.png", (90, 215, 2440, 1300)),
    "evidently": (IMG / "evidently-2026-08.png", (60, 240, 2500, 870)),
    "mlflow": (IMG / "mlflow-metricas.png", (0, 395, 2412, 1966)),
}


def make_crops():
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, (source, box) in CROPS.items():
        image = Image.open(source).convert("RGB").crop(box)
        image.save(OUT / "crops" / f"{name}.png", optimize=True)
        sizes[name] = image.size
    return sizes


# ------------------------------------------------------------------ quadros
CSS = """
:root{--ink:#14213d;--ink2:#46536e;--muted:#5f6b83;--page:#eef1f6;--card:#fff;--line:#d3d9e4;--blue:#2a78d6;
 --wash:#e3eefb;--alert:#e2622f;--crit:#d03b3b;--good:#0c8f0c;
 --display:"DIN Alternate","Bahnschrift","D-DIN",system-ui,sans-serif;
 --body:-apple-system,"SF Pro Text","Helvetica Neue",Arial,sans-serif;--mono:"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:var(--page);color:var(--ink);font-family:var(--body)}
.bar{position:absolute;top:0;left:0;right:0;height:92px;display:flex;align-items:center;justify-content:space-between;
 padding:0 64px;border-bottom:3px solid var(--ink)}
.brand{font-family:var(--display);font-size:36px;font-weight:700}
.brand span{font-family:var(--body);font-weight:400;font-size:24px;color:var(--ink2);margin-left:20px}
.star{display:flex;gap:8px;list-style:none;margin:0;padding:0;font-size:24px}
.star li{padding:8px 22px;border-radius:999px;color:var(--muted)}
.star li.done{color:var(--ink)}
.star li.on{background:var(--ink);color:#fff;font-weight:600}
.stage{position:absolute;top:95px;left:0;right:0;bottom:170px;padding:40px 64px}
.caption{position:absolute;left:0;right:0;bottom:0;height:170px;background:var(--ink);color:#fff;display:flex;
 align-items:center;justify-content:center;padding:0 120px}
.caption p{margin:0;font-size:39px;line-height:1.3;text-align:center;font-weight:500;text-wrap:balance}
h1,h2{font-family:var(--display);margin:0;font-weight:700;letter-spacing:.2px}
h2{font-size:54px;line-height:1.1}
.lead{font-size:30px;color:var(--ink2);margin:14px 0 0;line-height:1.35}
.dim{opacity:.2}
.center{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center}
.tag{font-size:26px;color:var(--ink2);margin:0 0 18px;align-self:flex-start}
.tag b{color:var(--ink)}
.shot{position:relative;display:inline-block;background:#fff;border-radius:14px;overflow:hidden;
 box-shadow:0 0 0 1px var(--line),0 10px 30px rgba(20,33,61,.10)}
.shot img{display:block}
.hl{position:absolute;border:6px solid var(--alert);border-radius:12px;box-shadow:0 0 0 3px #fff}
.note{font-size:28px;line-height:1.35;color:var(--ink);margin:22px 0 0;max-width:1560px;text-align:center}
.note b{font-weight:650}
.cols{display:grid;gap:48px;height:100%;align-items:center}
.card{background:var(--card);border-radius:14px;box-shadow:0 0 0 1px var(--line);padding:24px 30px}
.card h3{margin:0 0 6px;font-size:31px;font-weight:650}
.card p{margin:0;font-size:26px;line-height:1.35;color:var(--ink2)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.stats div{background:var(--card);border-radius:14px;box-shadow:0 0 0 1px var(--line);padding:18px 24px}
.stats b{display:block;font-family:var(--display);font-size:58px;line-height:1.05}
.stats span{font-size:24px;color:var(--ink2)}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-top:44px}
.step{background:var(--card);border-radius:14px;box-shadow:0 0 0 1px var(--line);padding:28px 28px 30px;min-height:330px}
.step i{font-family:var(--display);font-style:normal;font-size:72px;line-height:1;color:var(--blue)}
.step h3{margin:14px 0 10px;font-size:34px;line-height:1.15}
.step p{margin:0;font-size:25px;line-height:1.35;color:var(--ink2)}
.step.on{background:var(--ink);box-shadow:none}
.step.on i{color:#9cc5f6}.step.on h3{color:#fff}.step.on p{color:#d6deec}
.rules{list-style:none;margin:26px 0 0;padding:0;font-size:29px;line-height:1.3}
.rules li{padding:11px 0 11px 46px;position:relative;border-bottom:1px solid var(--line)}
.rules li::before{content:"";position:absolute;left:6px;top:19px;width:20px;height:11px;border-left:4px solid var(--good);
 border-bottom:4px solid var(--good);transform:rotate(-45deg)}
pre{margin:0;font-family:var(--mono);font-size:21px;line-height:1.45;white-space:pre;color:#e8edf6}
.code{background:#172544;border-radius:14px;padding:26px 30px;overflow:hidden}
.code .c{color:#93a4c4}.code .k{color:#9cc5f6}.code .s{color:#f3c98b}
.lv{display:grid;grid-template-columns:330px 1fr 330px;gap:0;background:var(--card);border-radius:14px;
 box-shadow:0 0 0 1px var(--line);margin-top:18px;align-items:center}
.lv>div{padding:20px 28px}
.lv h3{margin:0;font-size:32px}.lv h3 small{display:block;font-size:24px;font-weight:400;color:var(--ink2);margin-top:4px}
.lv p{margin:0;font-size:26px;line-height:1.4}.lv .keep{font-size:24px;color:var(--ink2)}
.lv b.n{font-family:var(--mono);font-size:24px;background:var(--wash);border-radius:6px;padding:2px 8px;font-weight:600}
.cal{display:grid;grid-template-columns:190px repeat(8,1fr);gap:6px;margin-top:22px;font-size:25px;text-align:center}
.cal div{background:var(--card);border-radius:8px;padding:9px 4px;box-shadow:0 0 0 1px var(--line)}
.cal .h{background:none;box-shadow:none;color:var(--ink2);font-size:23px}
.cal .l{text-align:left;padding-left:16px;font-weight:600}
.cal .t1,.cal .c1{background:#d6e6fa}.cal .t2,.cal .c2{background:#9ec5f4}.cal .t3,.cal .c3{background:#256abf;color:#fff}
.cal .x{color:var(--crit);font-size:21px;background:none;box-shadow:none;padding:2px}
.gov{display:grid;grid-template-columns:1fr 1fr;gap:36px;margin-top:30px}
.gov h3{font-size:34px;margin:0 0 6px}
.gov ul{margin:0;padding:0;list-style:none;font-size:28px;line-height:1.3}
.gov li{padding:13px 0;border-bottom:1px solid var(--line)}
.gov li b{font-weight:650}
.big{font-family:var(--display);font-size:96px;line-height:1.05;margin:0}
.mid{height:100%;display:flex;flex-direction:column;justify-content:center}
.three .card{padding:30px 34px}.three .card h3{font-size:36px;margin-bottom:10px}.three .card p{font-size:29px}
.list{list-style:none;margin:0;padding:0;font-size:29px;line-height:1.3}
.list li{padding:12px 0 12px 30px;border-bottom:1px solid var(--line);position:relative}
.list li::before{content:"";position:absolute;left:4px;top:25px;width:10px;height:10px;border-radius:50%;background:var(--blue)}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:40px}
.title{position:absolute;top:0;left:0;right:0;bottom:170px;background:var(--ink);color:#fff;padding:150px 120px 0}
.title .k{font-size:34px;color:#b9c7e0;margin:0 0 30px}
.title h1{font-size:104px;line-height:1.04;max-width:1500px}
.title .s{font-size:34px;color:#d6deec;margin:40px 0 0;max-width:1300px;line-height:1.35}
.title+.caption{background:#0d1730}
"""

STAGES = [("S", "Situação"), ("T", "Tarefa"), ("A", "Ação"), ("R", "Resultado")]
SIZES = {}


def shot(name, max_w, max_h, boxes=()):
    """Captura recortada, com realces opcionais dados em pixels do recorte."""
    w, h = SIZES[name]
    scale = min(max_w / w, max_h / h)
    marks = "".join(
        f'<div class="hl" style="left:{x0 * scale - 8:.0f}px;top:{y0 * scale - 8:.0f}px;'
        f'width:{(x1 - x0) * scale + 16:.0f}px;height:{(y1 - y0) * scale + 16:.0f}px"></div>'
        for x0, y0, x1, y1 in boxes)
    return (f'<div class="shot"><img src="file://{OUT}/crops/{name}.png" width="{w * scale:.0f}" '
            f'height="{h * scale:.0f}">{marks}</div>')


def dim(active):
    return "" if active else " dim"


def scene_titulo(step):
    return ('<div class="title"><p class="k">Tech Challenge, Fase 4 &nbsp;|&nbsp; MLET/FIAP</p>'
            "<h1>Drift, observabilidade e governança de um modelo de crédito</h1>"
            '<p class="s">Contrato de dados, experimentos de drift, monitoramento por lote e proposta de governança.</p></div>')


def scene_situacao(step):
    facts = [
        ("Uma fintech e um modelo de pontuação de crédito", "O modelo estima o risco de inadimplência de cada cliente."),
        ("O risco: degradação silenciosa", "A economia e o perfil dos clientes mudam, e o modelo continua respondendo."),
        ("Base Give Me Some Credit", "150.000 registros, 10 variáveis de entrada. Modelo de referência: XGBoost."),
    ]
    cards = "".join(f'<div class="card{dim(step >= i)}" style="margin-bottom:18px"><h3>{t}</h3><p>{d}</p></div>'
                    for i, (t, d) in enumerate(facts, start=1))
    stats = (f'<div class="stats{dim(step >= 4)}"><div><b>0,857</b><span>AUC na validação</span></div>'
             "<div><b>77%</b><span>dos inadimplentes identificados (recall)</span></div>"
             "<div><b>0,568</b><span>KS do modelo</span></div></div>")
    return ('<div class="cols" style="grid-template-columns:660px 1fr"><div><h2>Um modelo de crédito pode piorar '
            'sem que ninguém perceba</h2><p class="lead">É esse o cenário do desafio: detectar a degradação antes '
            f"que ela vire prejuízo.</p></div><div>{cards}{stats}</div></div>")


def scene_tarefa(step):
    items = [("Validar os dados", "Contrato de dados em Pandera. O lote inválido não chega ao modelo."),
             ("Simular e detectar drift", "Data drift e concept drift separados. Evidently, KS e PSI."),
             ("Observar o processamento", "Log central, MLflow, regras de alerta e dashboard."),
             ("Documentar a governança", "LGPD, vieses, causalidade e resposta à degradação.")]
    tiles = "".join(
        f'<div class="step{" on" if step == i else ""}{dim(step == 0 or step >= i)}"><i>{i}</i><h3>{t}</h3><p>{d}</p></div>'
        for i, (t, d) in enumerate(items, start=1))
    return f'<div class="mid"><h2>A tarefa, em quatro etapas</h2><div class="steps">{tiles}</div></div>'


def scene_contrato(step):
    rules = ["Colunas e tipos definidos; coluna extra reprova", "Alvo binário e valores finitos",
             "Idade inteira entre 18 e 100", "Renda obrigatória e não negativa",
             "Medidas e contagens não negativas", "Linha duplicada ou lote vazio reprova"]
    code = (
        '<span class="c"># src/contract.py</span>\n'
        'columns = {\n'
        '    TARGET: pa.Column(int, pa.Check.isin([<span class="s">0</span>, <span class="s">1</span>])),\n'
        '    <span class="s">"age"</span>: pa.Column(int, pa.Check.in_range(<span class="s">18</span>, <span class="s">100</span>)),\n'
        '    <span class="s">"MonthlyIncome"</span>: pa.Column(float, pa.Check.ge(<span class="s">0</span>),\n'
        '                               nullable=<span class="k">False</span>),\n'
        '    <span class="s">"DebtRatio"</span>: pa.Column(float, pa.Check.ge(<span class="s">0</span>)),\n'
        '    ...\n'
        '}\n'
        'schema = pa.DataFrameSchema(\n'
        '    columns,\n'
        '    checks=pa.Check(<span class="k">lambda</span> df: <span class="k">not</span> df.empty),\n'
        '    strict=<span class="k">True</span>,          <span class="c"># coluna inesperada reprova</span>\n'
        '    unique=list(columns),  <span class="c"># linha duplicada reprova</span>\n'
        ')')
    return ('<div class="cols" style="grid-template-columns:1fr 880px"><div><h2>Contrato de dados</h2>'
            '<p class="lead">Escrito em Pandera. Vale para o treino e para cada lote que chega.</p>'
            f'<ul class="rules">{"".join(f"<li>{r}</li>" for r in rules)}</ul></div>'
            f'<div class="code"><pre>{code}</pre></div></div>')


def scene_bloqueio(step):
    if step == 1:
        boxes = [(1262, 8, 1470, 617)]
        note = ("<b>Entrega corrompida de maio:</b> 400 linhas duplicadas, 120 rendas nulas, 40 idades fora de 18 a 100 "
                "e 20 razões de dívida negativas. As 4.612 linhas foram para a quarentena.")
    else:
        boxes = [(400, 116, 2330, 216)]
        note = "<b>Os outros oito lotes passaram no contrato</b>, inclusive os que têm drift: são dados válidos."
    return ('<div class="center"><p class="tag"><b>Dashboard do projeto</b>: sinais por lote, na ordem de chegada</p>'
            f'{shot("sinais", 1792, 520, boxes)}<p class="note">{note}</p></div>')


def scene_niveis(step):
    data = (f'<div class="lv{dim(step in (1, 2))}"><div><h3>Data drift<small>muda P(X)</small></h3></div>'
            '<div><p>Renda <b class="n">×0,90</b> <b class="n">×0,75</b> <b class="n">×0,60</b><br>'
            'Endividamento e uso do rotativo <b class="n">×1,10</b> <b class="n">×1,30</b> <b class="n">×1,60</b><br>'
            "Quem passa a usar todo o limite ganha um atraso de 30 a 59 dias</p></div>"
            '<div><p class="keep">Níveis t1, t2 e t3.<br>Rótulos preservados.</p></div></div>')
    concept = (f'<div class="lv{dim(step in (1, 3))}"><div><h3>Concept drift<small>muda P(Y|X)</small></h3></div>'
               '<div><p>No quarto de menor renda, um bom pagador passa a inadimplir com chance de '
               '<b class="n">0,10</b> <b class="n">0,25</b> <b class="n">0,50</b></p></div>'
               '<div><p class="keep">Níveis c1, c2 e c3.<br>Variáveis e scores preservados.</p></div></div>')
    months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago"]
    t = ["t0", "t0", "t0", "t1", "t1", "t2", "t2", "t3"]
    c = ["c0", "c0", "c0", "c0", "c1", "c1", "c2", "c3"]
    cal = ('<div class="h"></div>' + "".join(f'<div class="h">{m}/26</div>' for m in months)
           + '<div class="l">Data drift</div>' + "".join(f'<div class="{v}">{v}</div>' for v in t)
           + '<div class="l">Concept drift</div>' + "".join(f'<div class="{v}">{v}</div>' for v in c))
    return ('<div class="mid"><h2>Os mesmos níveis de drift em todos os experimentos</h2>'
            f'{data}{concept}<div class="cal{dim(step in (1, 4))}">{cal}</div>'
            f'<p class="lead{dim(step == 4)}" style="font-size:26px;margin-top:14px">Oito lotes mensais seguem esse calendário; '
            "em maio chega também uma entrega corrompida.</p></div>")


def scene_evidently(step):
    return ('<div class="center"><p class="tag"><b>Relatório do Evidently</b>, lote de agosto contra a referência de treino</p>'
            f'{shot("evidently", 1792, 462)}'
            '<p class="note">Com drift: renda, endividamento e utilização do rotativo (teste KS, p &lt; 0,01). '
            "As contagens e a idade usam PSI de categorias, com limite de 0,20.</p></div>")


def scene_efeito(step):
    boxes = [(535, 70, 760, 660)] if step == 1 else [(785, 215, 935, 285), (785, 512, 935, 582)]
    note = ("O Evidently diz <b>se</b> há drift: três features nos três cenários. O PSI por decis diz <b>de que tamanho</b>."
            if step == 1 else
            "PSI da renda: <b>0,030</b> em t1 e <b>0,600</b> em t3. O veredito do teste é o mesmo; o tamanho, não.")
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: cenários pareados, com os mesmos 35.297 clientes</p>'
            f'{shot("cenarios", 1792, 512, boxes)}<p class="note">{note}</p></div>')


def scene_pipeline(step):
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: saúde do pipeline, uma linha por lote</p>'
            f'{shot("pipeline", 1792, 590)}</div>')


def scene_logmlflow(step):
    event = (
        '{\n'
        '  <span class="k">"run_id"</span>: <span class="s">"mon-20261115T024544Z-cd2bd7"</span>,\n'
        '  <span class="k">"stage"</span>: <span class="s">"drift"</span>,\n'
        '  <span class="k">"batch"</span>: <span class="s">"2026-08"</span>,\n'
        '  <span class="k">"status"</span>: <span class="s">"success"</span>,\n'
        '  <span class="k">"duration_s"</span>: 0.4938,\n'
        '  <span class="k">"rows"</span>: 4412,\n'
        '  <span class="k">"details"</span>: {\n'
        '    <span class="k">"max_psi"</span>: 0.612,\n'
        '    <span class="k">"alert"</span>: [<span class="s">"MonthlyIncome"</span>, <span class="s">"DebtRatio"</span>]\n'
        '  }\n'
        '}')
    return ('<div class="cols" style="grid-template-columns:740px 1fr;align-items:center">'
            '<div><p class="tag"><b>Log central</b>: um evento por etapa de cada lote</p>'
            f'<div class="code"><pre style="font-size:22px">{event}</pre></div>'
            '<p class="lead" style="font-size:26px">56 eventos nesta execução, todos com o mesmo identificador.</p></div>'
            '<div><p class="tag"><b>MLflow</b>: 25 métricas por lote, em série</p>'
            f'{shot("mlflow", 1000, 640)}</div></div>')


def scene_painel(step):
    parts = ["Veredito do último lote e sinais por lote", "AUC e PSI na mesma linha do tempo",
             "Saúde do pipeline: status, volume e duração", "Mapa de PSI por variável e por lote",
             "Cenários pareados e desempenho do modelo", "Alertas com filtro por severidade",
             "Atribuição causal e definição das métricas"]
    return ('<div class="cols" style="grid-template-columns:1010px 1fr"><div>'
            f'{shot("painel", 1010, 660)}</div><div><h2>Um painel, um arquivo</h2>'
            '<p class="lead">reports/dashboard.html abre direto no navegador, sem servidor.</p>'
            f'<ul class="list" style="margin-top:22px">{"".join(f"<li>{x}</li>" for x in parts)}</ul></div></div>')


def scene_governanca(step):
    left = ["<b>Base legal proposta:</b> proteção do crédito (LGPD, art. 7º, X)",
            "<b>Minimização:</b> o contrato recusa colunas extras",
            "<b>Retenção:</b> de 30 dias (quarentena) a 12 meses (relatórios e modelos)",
            "<b>Logs, resumos e dashboard</b> só com valores agregados"]
    right = ["<b>Vieses:</b> avaliação por segmento antes de qualquer uso operacional",
             "<b>Causalidade:</b> hipóteses e a evidência que cada uma exige",
             "<b>Resposta:</b> registrar o incidente, análise humana e retreino só com comparação"]
    return ('<div class="mid"><h2>Governança: o que está implementado e o que é proposta</h2><div class="gov">'
            f'<div class="card"><h3>Proteção de dados</h3><ul>{"".join(f"<li>{i}</li>" for i in left)}</ul></div>'
            f'<div class="card{dim(step >= 2)}"><h3>Uso responsável e resposta</h3><ul>{"".join(f"<li>{i}</li>" for i in right)}</ul></div>'
            "</div></div>")


def scene_tempo(step):
    if step == 1:
        return ('<div class="center"><p class="tag"><b>Dashboard</b>: AUC por lote mensal, contra o baseline da validação</p>'
                f'{shot("auc", 1792, 470)}'
                '<p class="note">De <b>0,854</b> em janeiro para <b>0,618</b> em agosto.</p></div>')
    note = ("Maio: AUC de <b>0,773</b>, com o maior PSI das variáveis em <b>0,037</b>." if step == 2 else
            "O concept drift não mexe em nenhuma entrada. Sem o rótulo, essa queda não aparece.")
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: desempenho (acima) e drift das entradas (abaixo), '
            f'na mesma linha do tempo</p>{shot("auc_psi", 1792, 560, [(1215, 20, 1385, 1100)])}'
            f'<p class="note">{note}</p></div>')


def scene_decisoes(step):
    boxes = [(1855, 70, 1995, 660)] if step == 1 else [(1990, 70, 2160, 660)]
    note = ("AUC de <b>0,843</b> para <b>0,829</b>: a ordenação dos clientes quase não muda." if step == 1 else
            "Sinalizados: de <b>24,8%</b> para <b>38,5%</b>, com a mesma inadimplência (6,6%).")
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: cenários pareados de data drift, rótulos preservados</p>'
            f'{shot("cenarios", 1792, 512, boxes)}<p class="note">{note}</p></div>')


def scene_causal(step):
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: experimento contrafactual no último lote '
            f'(mesmos 4.412 clientes)</p>{shot("causal", 1792, 440)}'
            '<p class="note">Só os rótulos: <b>−0,181</b> de AUC. Só os dados: <b>−0,010</b>, mas a parcela sinalizada '
            "vai de 24,4% para 38,1%.</p></div>")


def scene_alertas(step):
    return ('<div class="center"><p class="tag"><b>Dashboard</b>: 21 alertas na execução (1 crítico, 13 alertas e 7 avisos)</p>'
            f'{shot("alertas", 1792, 620)}</div>')


def scene_limites(step):
    items = ["Os lotes de produção são sintéticos, feitos com clientes que o modelo nunca viu.",
             "Os níveis de drift são intervenções conhecidas, não estimativas econômicas.",
             "O que fica demonstrado é o funcionamento do monitoramento."]
    cards = "".join(f'<div class="card"><p style="font-size:32px;line-height:1.35;color:var(--ink)">{i}</p></div>' for i in items)
    return f'<div class="mid"><p class="big">Tudo isso é simulação.</p><div class="three">{cards}</div></div>'


def scene_conclusao(step):
    items = [("Rótulos maturados", "Só eles revelam o concept drift."),
             ("Decisões", "Parcela sinalizada e PSI do score se movem antes dos limites das variáveis."),
             ("Desempenho por segmento", "A queda pode se concentrar em um grupo de clientes.")]
    cards = "".join(f'<div class="card"><h3>{t}</h3><p>{d}</p></div>' for t, d in items)
    tail = ('<p class="lead" style="margin-top:36px;font-size:32px{}">Código, relatórios e documentação estão no '
            "repositório do projeto.</p>").format("" if step == 2 else ";visibility:hidden")
    return f'<div class="mid"><p class="big">Vigiar as entradas não basta.</p><div class="three">{cards}</div>{tail}</div>'


SCENES = {name[6:]: fn for name, fn in list(globals().items()) if name.startswith("scene_")}


def frame_html(stage, scene, step, caption):
    body = SCENES[scene](step)
    cap = f'<footer class="caption"><p>{html.escape(caption)}</p></footer>'
    if scene == "titulo":
        return f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{body}{cap}"
    order = [key for key, _ in STAGES]
    items = "".join(
        f'<li class="{"on" if key == stage else "done" if order.index(key) < order.index(stage) else ""}">{label}</li>'
        for key, label in STAGES)
    bar = ('<header class="bar"><div class="brand">TC04<span>Drift, observabilidade e governança</span></div>'
           f'<ol class="star">{items}</ol></header>')
    return f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{bar}<main class='stage'>{body}</main>{cap}"


def render_frames():
    SIZES.update(make_crops())
    frames = OUT / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for index, (stage, scene, step, caption, _) in enumerate(SCRIPT, start=1):
        page, png = frames / f"f{index:02d}.html", frames / f"f{index:02d}.png"
        page.write_text(frame_html(stage, scene, step, caption), encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1920,1080", f"--screenshot={png}", f"file://{page}"], check=True, capture_output=True)
        size = Image.open(png).size
        assert size == (1920, 1080), (png, size)
    print(f"{len(SCRIPT)} quadros em {frames}")


# ------------------------------------------------------------------ narração
def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == SR and handle.getnchannels() == 1 and handle.getsampwidth() == 2
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def trim(samples, threshold=350, keep=0.06):
    """Remove o silêncio das pontas, deixando uma pequena margem."""
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > threshold)
    if not len(loud):
        return samples
    pad = int(keep * SR)
    return samples[max(loud[0] - pad, 0):min(loud[-1] + pad, len(samples))]


def render_audio():
    ffmpeg, voice_dir = ffmpeg_exe(), OUT / "voz"
    voice_dir.mkdir(parents=True, exist_ok=True)
    pieces, timeline, cursor = [np.zeros(int(LEAD_IN * SR), dtype=np.int16)], [], LEAD_IN
    for index, (stage, scene, step, caption, spoken) in enumerate(SCRIPT, start=1):
        aiff, wav = voice_dir / f"s{index:02d}.aiff", voice_dir / f"s{index:02d}.wav"
        spoken = spoken or caption
        for pattern, replacement in PRONUNCIA:
            spoken = re.sub(pattern, replacement, spoken)
        subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), spoken], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(SR), "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        speech = trim(read_wav(wav))
        last = index == len(SCRIPT)
        nxt = None if last else SCRIPT[index]
        gap = TAIL if last else GAP_STAGE if nxt[0] != stage else GAP_SCENE if nxt[1] != scene else GAP
        start, end = cursor, cursor + len(speech) / SR
        pieces += [speech, np.zeros(int(gap * SR), dtype=np.int16)]
        cursor = end + int(gap * SR) / SR
        timeline.append({"n": index, "stage": stage, "scene": scene, "step": step, "caption": caption,
                         "spoken": spoken, "start": round(start, 3), "end": round(end, 3)})
    audio = np.concatenate(pieces)
    with wave.open(str(OUT / "narracao.wav"), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes(audio.tobytes())
    total = len(audio) / SR
    # cada quadro fica na tela do início da sua fala até o início da próxima
    for item, following in zip(timeline, timeline[1:] + [None]):
        item["show_from"] = 0.0 if item["n"] == 1 else item["start"]
        item["show_until"] = round(total, 3) if following is None else following["start"]
    (OUT / "timeline.json").write_text(json.dumps({"total": round(total, 3), "items": timeline}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    words = sum(len(item["caption"].split()) for item in timeline)
    print(f"narração: {total:.1f} s ({int(total // 60)}:{total % 60:04.1f}), {len(timeline)} frases, {words} palavras")
    return timeline, total


# ------------------------------------------------------------------ vídeo, legendas e roteiro
def stamp(seconds, sep=","):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def render_video():
    ffmpeg = ffmpeg_exe()
    data = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    items, total = data["items"], data["total"]
    lines = ["ffconcat version 1.0"]
    for item in items:
        lines += [f"file 'frames/f{item['n']:02d}.png'", f"duration {item['show_until'] - item['show_from']:.3f}"]
    lines.append(f"file 'frames/f{items[-1]['n']:02d}.png'")
    (OUT / "quadros.ffconcat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    target = OUT / "tc04-star.mp4"
    fade = f"fade=t=in:st=0:d=0.4,fade=t=out:st={total - 0.6:.2f}:d=0.6"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(OUT / "quadros.ffconcat"),
                    "-i", str(OUT / "narracao.wav"),
                    "-vf", f"fps=30,format=yuv420p,{fade}", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-tune", "stillimage", "-g", "300",
                    "-c:a", "aac", "-b:a", "112k", "-ac", "1", "-t", f"{total:.3f}", "-movflags", "+faststart", str(target)],
                   check=True, cwd=OUT)
    srt = []
    for item in items:
        srt += [str(item["n"]), f"{stamp(item['start'])} --> {stamp(item['end'] + 0.15)}", item["caption"], ""]
    (OUT / "tc04-star.srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"vídeo: {target} ({target.stat().st_size / 1e6:.1f} MB), {stamp(total, '.')[3:]}")


VISUAL = {
    "titulo": "Cartela de título", "situacao": "Quadro da situação: contexto, base de dados e métricas do baseline",
    "tarefa": "As quatro etapas da tarefa", "contrato": "Regras do contrato e trecho de src/contract.py",
    "bloqueio": "Dashboard: sinais por lote, com a entrega corrompida de maio em destaque",
    "niveis": "Níveis de data drift e de concept drift e o calendário dos lotes mensais",
    "evidently": "Relatório do Evidently do lote de agosto",
    "efeito": "Dashboard: cenários pareados, veredito do Evidently e PSI por variável",
    "pipeline": "Dashboard: saúde do pipeline por lote", "logmlflow": "Evento do log central e métricas no MLflow",
    "painel": "Visão geral do dashboard e lista das seções",
    "governanca": "Quadro de governança: proteção de dados, uso responsável e resposta",
    "tempo": "Dashboard: AUC e PSI por lote, com maio em destaque",
    "decisoes": "Dashboard: cenários pareados, colunas de AUC e de clientes sinalizados",
    "causal": "Dashboard: experimento contrafactual", "alertas": "Dashboard: tabela de alertas",
    "limites": "Quadro de limites: tudo é simulação", "conclusao": "Quadro de conclusão",
}
STAGE_NAME = {"": "Abertura", "S": "Situation (situação)", "T": "Task (tarefa)", "A": "Action (ação)", "R": "Result (resultado)"}


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def render_script():
    data = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    items, total = data["items"], data["total"]
    blocks = []
    for item in items:
        if not blocks or blocks[-1]["stage"] != item["stage"]:
            blocks.append({"stage": item["stage"], "from": item["show_from"], "items": []})
        blocks[-1]["items"].append(item)
    for block, following in zip(blocks, blocks[1:] + [None]):
        block["until"] = total if following is None else following["from"]
    lines = ["# Roteiro do vídeo STAR", "",
             f"Vídeo: [tc04-star.mp4](tc04-star.mp4), {clock(total)} de duração (limite de 5 minutos), 1920×1080. "
             "Legendas: [tc04-star.srt](tc04-star.srt); as mesmas frases estão gravadas na imagem.", "",
             "A narração é sintetizada (voz Luciana, pt-BR, do macOS) e os integrantes não aparecem. Sem poder ouvir o áudio, a "
             "pronúncia foi conferida passando cada frase por um reconhecedor de fala e comparando com o roteiro; a grafia "
             "usada só para a fala foi ajustada onde ele não entendia. Os quadros usam as capturas "
             "de `docs/img/` e números agregados dos relatórios versionados; nenhum dado individual de cliente é exibido.", "",
             "| Bloco | Intervalo | Duração |", "| --- | --- | ---: |"]
    lines += [f"| {STAGE_NAME[b['stage']]} | {clock(b['from'])} a {clock(b['until'])} | {b['until'] - b['from']:.0f} s |" for b in blocks]
    for block in blocks:
        lines += ["", f"## {STAGE_NAME[block['stage']]}", "", "| Início | Na tela | Narração |", "| ---: | --- | --- |"]
        seen = None
        for item in block["items"]:
            visual = VISUAL[item["scene"]] if item["scene"] != seen else "(mesmo quadro)"
            seen = item["scene"]
            lines.append(f"| {clock(item['start'])} | {visual} | {item['caption']} |")
    lines += ["", "## Como regenerar", "",
              "O vídeo é montado por [gera_video.py](gera_video.py): cada frase vira um quadro (HTML renderizado pelo Chrome) "
              "e um trecho de fala (comando `say`), e o ffmpeg junta tudo. Para trocar uma frase, editar a lista `SCRIPT` e rodar:",
              "", "```bash", "python docs/video/gera_video.py . /tmp/tc04-video tudo",
              "cp /tmp/tc04-video/tc04-star.mp4 /tmp/tc04-video/tc04-star.srt /tmp/tc04-video/roteiro.md docs/video/", "```", "",
              "Requer macOS com a voz Luciana, Google Chrome, Pillow, NumPy e ffmpeg. Na lista, o quinto campo de cada frase é a "
              "grafia usada só para a fala (siglas e termos em inglês escritos como se pronunciam). Para gravar a narração com "
              "voz própria, tirar o áudio com `ffmpeg -i tc04-star.mp4 -an -c:v copy sem-audio.mp4` e narrar sobre as legendas.", ""]
    (OUT / "roteiro.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"roteiro: {OUT / 'roteiro.md'}")


if __name__ == "__main__":
    if MODE in ("quadros", "tudo"):
        render_frames()
    if MODE in ("audio", "tudo"):
        render_audio()
    if MODE in ("video", "tudo"):
        render_video()
    if MODE in ("roteiro", "tudo"):
        render_script()
