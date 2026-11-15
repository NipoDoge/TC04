"""Estilo e script embutidos no dashboard (ver ``src.dashboard``).

Identidade: boletim técnico de monitoramento. Tinta azul-marinho sobre folha
branca, um trilho lateral com os dados da execução e o índice, seções abertas
separadas por um fio (sem cartões) e uma face DIN, de painéis de instrumentos,
nos títulos e nos números. Nada é carregado da internet: as fontes são as do
sistema, com alternativas.

Cores por papel, com valores próprios para os modos claro e escuro: três séries
categóricas (validadas para daltonismo nas duas superfícies), uma rampa
sequencial de um só matiz para o PSI, o par divergente para efeitos positivos e
negativos e as cores de status, que sempre aparecem com ícone e rótulo. Os tons
de texto têm contraste de pelo menos 4,5:1 sobre a folha e sobre o trilho.
"""

DARK = """color-scheme:dark;
  --page:#0a101c;--surface-1:#0f1626;--text-primary:#e9eef7;--text-secondary:#b6c1d6;--muted:#8f9bb3;
  --grid:#1e2940;--axis:#33415c;--rule:#e9eef7;--border:rgba(233,238,247,.14);--link:#86b6ef;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--neg:#e66767;--pos:#3987e5;
  --seq-1:#0d366b;--seq-2:#184f95;--seq-3:#256abf;--seq-4:#5598e7;--seq-5:#9ec5f4;
  --seq-1-ink:#fff;--seq-2-ink:#fff;--seq-3-ink:#fff;--seq-4-ink:#0f1626;--seq-5-ink:#0f1626;--track:#0d366b;
  --cell:rgba(233,238,247,.06);--wash-warning:rgba(250,178,25,.20);--wash-serious:rgba(236,131,90,.24);
  --wash-critical:rgba(208,59,59,.30);--shadow:rgba(0,0,0,.5)"""

CSS = """
.viz-root{color-scheme:light;
  --page:#eef1f6;--surface-1:#ffffff;--text-primary:#14213d;--text-secondary:#46536e;--muted:#5f6b83;
  --grid:#e4e8ef;--axis:#c2cad8;--rule:#14213d;--border:rgba(20,33,61,.14);--link:#1c5cab;
  --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--neg:#e34948;--pos:#2a78d6;
  --seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#5598e7;--seq-4:#256abf;--seq-5:#104281;
  --seq-1-ink:#14213d;--seq-2-ink:#14213d;--seq-3-ink:#14213d;--seq-4-ink:#fff;--seq-5-ink:#fff;
  --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--track:#cde2fb;
  --cell:rgba(20,33,61,.05);--wash-warning:rgba(250,178,25,.22);--wash-serious:rgba(236,131,90,.24);
  --wash-critical:rgba(208,59,59,.16);--shadow:rgba(20,33,61,.18);
  --display:"DIN Alternate","Bahnschrift","D-DIN","Barlow","Roboto Condensed",system-ui,sans-serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .viz-root{%DARK%}}
:root[data-theme="dark"] .viz-root{%DARK%}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
.viz-root{margin:0;background:var(--surface-1);color:var(--text-primary);font:14px/1.55 var(--body)}
p,h1,h2,h3,dl,dd,ul{margin:0}
ul{padding:0;list-style:none}
a{color:var(--link);text-underline-offset:2px}
code{font:inherit;color:var(--text-secondary);overflow-wrap:anywhere}
:focus-visible{outline:2px solid var(--series-1);outline-offset:2px;border-radius:2px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.skip{position:absolute;left:8px;top:-48px;z-index:20;padding:8px 14px;background:var(--text-primary);color:var(--surface-1);border-radius:4px}
.skip:focus{top:8px}
.muted{color:var(--muted)}.strong{font-weight:600}.num{font-variant-numeric:tabular-nums}

/* estrutura: trilho + coluna principal */
.layout{display:grid;grid-template-columns:252px minmax(0,1fr);min-height:100vh}
.rail{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:30px 22px 24px;
  background:var(--page);border-right:1px solid var(--border);display:flex;flex-direction:column;gap:26px}
.product{font:700 17px/1.25 var(--display);letter-spacing:.01em}
.product-sub{font-size:12.5px;color:var(--text-secondary);margin-top:2px}
.run dt{font-size:12px;color:var(--muted)}
.run dd{font-size:13px;margin-bottom:10px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.toc a{display:block;padding:5px 0 5px 12px;border-left:2px solid var(--axis);color:var(--text-secondary);
  text-decoration:none;font-size:13.5px}
.toc a:hover,.toc a:focus-visible,.toc a[aria-current]{border-left-color:var(--text-primary);color:var(--text-primary)}
.toc a[aria-current]{font-weight:600}
.theme{margin-top:auto;align-self:flex-start;font:inherit;font-size:12.5px;padding:6px 12px;border-radius:4px;
  border:1px solid var(--muted);background:transparent;color:var(--text-primary);cursor:pointer}
.theme:hover{border-color:var(--text-primary)}
.main{min-width:0;max-width:1260px;padding:38px clamp(20px,4vw,60px) 80px}

/* abertura */
h1{font:700 34px/1.1 var(--display);letter-spacing:-.005em;max-width:22ch}
.verdict{margin-top:18px;padding:4px 0 4px 16px;border-left:4px solid var(--muted);font-size:17px;line-height:1.45;max-width:72ch}
.verdict .chip{font-weight:700}
.v-good{border-left-color:var(--good)}.v-warning{border-left-color:var(--warning)}
.v-serious{border-left-color:var(--serious)}.v-critical{border-left-color:var(--critical)}
.synthetic{margin-top:12px;color:var(--text-secondary);font-size:13px;max-width:78ch}

/* quadro de sinais por lote */
.board{width:100%;margin-top:30px;border-collapse:separate;border-spacing:3px;table-layout:fixed}
.board caption{caption-side:top;text-align:left;padding:0 0 10px 3px;font-size:13px;color:var(--text-secondary)}
.board col.signal{width:190px}
.board thead th{padding:2px 2px 6px;text-align:center;vertical-align:bottom;font:700 14px/1.2 var(--display)}
.board thead th small{display:block;font:400 11px/1.25 var(--body);color:var(--muted)}
.board tbody th{padding:0 12px 0 3px;text-align:left;font-weight:500;font-size:13.5px;vertical-align:middle}
.board td{height:46px;padding:2px;text-align:center;vertical-align:middle;border-radius:4px;background:var(--cell);
  font-size:12.5px;line-height:1.2;cursor:default}
.board td .ico{display:block;font-size:13px}
.board td.s-ok .ico{color:var(--muted);font-weight:400}
.board td.s-warning{background:var(--wash-warning)}
.board td.s-alert{background:var(--wash-serious)}
.board td.s-critical{background:var(--wash-critical)}
.board td.s-na{background:transparent;color:var(--muted)}
.board td:hover{box-shadow:inset 0 0 0 1px var(--text-secondary)}

/* fatos da execução */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));margin-top:30px;
  border-top:2px solid var(--rule);border-bottom:1px solid var(--axis)}
.stat{padding:14px 18px 14px 0}
.stat-label{font-size:12.5px;color:var(--text-secondary)}
.stat-value{font:700 28px/1.2 var(--display)}
.stat-note{font-size:12px;color:var(--text-secondary)}
.spark{display:block;width:120px;height:30px;margin-top:6px}
.spark path{fill:none;stroke:var(--axis);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.spark circle{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}

/* seções */
.block{margin-top:60px;padding-top:16px;border-top:2px solid var(--rule);scroll-margin-top:16px}
h2{font:700 22px/1.2 var(--display);letter-spacing:.005em}
h3{font:600 13.5px/1.4 var(--body);margin:28px 0 8px}
.sub{margin-top:4px;color:var(--text-secondary);font-size:13.5px;max-width:78ch}
.reading{margin-top:14px;font-size:15px;max-width:76ch}
.note{margin-top:10px;color:var(--text-secondary);font-size:12.5px;max-width:92ch}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:8px 36px}
.grid2>div{min-width:0}

/* status: cor sempre com ícone e rótulo */
.chip{display:inline-flex;align-items:baseline;gap:5px;white-space:nowrap}
.ico{font-size:12px;font-weight:700}
.ico-good{color:var(--good)}.ico-warning{color:var(--warning)}.ico-serious{color:var(--serious)}
.ico-critical{color:var(--critical)}.ico-neutral{color:var(--muted)}

/* gráficos */
.chart{display:block;width:100%;height:auto;overflow:visible}
.chart text{font-family:var(--body)}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.ref{stroke:var(--text-secondary);stroke-width:1;stroke-dasharray:5 4}
.ref-label{fill:var(--text-secondary);font-size:11.5px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface-1);stroke-width:2}
.end,.chart .note,.ref-label{paint-order:stroke;stroke:var(--surface-1);stroke-width:4px;stroke-linejoin:round}
.end{fill:var(--text-primary);font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums}
.note-ring{fill:none;stroke:var(--text-primary);stroke-width:1.5}
.chart .note{fill:var(--text-primary);font-size:12px;font-weight:600}
.crosshair{stroke:var(--axis);stroke-width:1;visibility:hidden}
.bar-label{fill:var(--text-primary);font-size:12.5px}
.bar{cursor:default}.bar:hover{filter:brightness(1.12)}
.whisker{fill:none;stroke:var(--text-primary);stroke-width:1.25}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--text-secondary);margin:4px 0 2px}
.key{display:inline-flex;align-items:center;gap:6px}
.key-line{width:16px;height:3px;border-radius:2px}
.swatch{width:14px;height:14px;border-radius:3px}

/* tabelas */
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 14px 8px 0;vertical-align:top}
.data thead th{font-size:12px;font-weight:600;color:var(--text-secondary);border-bottom:1px solid var(--text-primary);white-space:nowrap}
.data tbody tr{border-bottom:1px solid var(--grid)}
.data td{font-variant-numeric:tabular-nums}
.data th.r,.data td.r{text-align:right}
.data tbody tr:hover{background:var(--cell)}
.health td{white-space:nowrap}.health .chip{margin-right:2px}
.meter,.inline-bar{display:inline-block;vertical-align:middle;height:8px;border-radius:4px;margin-left:8px}
.meter{width:90px;background:var(--track)}
.meter span{display:block;height:100%;border-radius:4px;background:var(--series-1)}
.inline-bar{width:64px}
.inline-bar span{display:block;height:100%;border-radius:0 4px 4px 0;background:var(--pos)}
.inline-bar.neg span{background:var(--neg);margin-left:auto;border-radius:4px 0 0 4px}
.causal td{white-space:nowrap}.causal td:first-child{white-space:normal}
.alerts td:nth-child(4){min-width:320px}
.defs td:first-child{font-weight:600;white-space:nowrap}

/* mapa de PSI */
.heat{border-collapse:separate;border-spacing:2px;margin-top:12px}
.heat caption{caption-side:top;text-align:left;padding:0 0 8px;font-size:13px;color:var(--text-secondary)}
.heat thead th{text-align:center;padding:4px 6px;font:700 13.5px/1.2 var(--display)}
.heat thead th:first-child{text-align:left;font:600 12px var(--body);color:var(--text-secondary)}
.heat thead a{display:block;min-height:24px;padding-top:4px;font:400 12px var(--body)}
.heat tbody th{font-weight:500;padding:6px 12px 6px 0;min-width:230px}
.heat tbody th code{display:block;font-size:11.5px;color:var(--muted)}
.heat td{text-align:center;padding:9px 6px;border-radius:3px;font-variant-numeric:tabular-nums;min-width:70px;cursor:default}
.heat td span{margin-right:4px;font-size:10px}
.heat td:hover{filter:brightness(1.08)}
.heat .score-row th,.heat .score-row td{border-top:8px solid var(--surface-1)}
.tag{margin-left:8px;padding:1px 7px;border:1px solid var(--axis);border-radius:9px;font-size:11px;color:var(--text-secondary);white-space:nowrap}
.seq-1{background:var(--seq-1);color:var(--seq-1-ink)}.seq-2{background:var(--seq-2);color:var(--seq-2-ink)}
.seq-3{background:var(--seq-3);color:var(--seq-3-ink)}.seq-4{background:var(--seq-4);color:var(--seq-4-ink)}
.seq-5{background:var(--seq-5);color:var(--seq-5-ink)}
.scale{margin-top:10px}
.pill{display:inline-block;min-width:64px;padding:2px 8px;border-radius:3px;text-align:center}
.factors{display:block;font-size:12px;color:var(--text-secondary);white-space:normal;max-width:30ch}

/* controles */
.filters{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:14px 0 8px}
.filters button{font:inherit;font-size:13px;padding:5px 13px;border-radius:4px;border:1px solid var(--muted);
  background:transparent;color:var(--text-primary);cursor:pointer}
.filters button:hover{border-color:var(--text-primary)}
.filters button[aria-pressed="true"]{background:var(--text-primary);color:var(--surface-1);border-color:var(--text-primary)}
.filters output{margin-left:6px;font-size:12.5px;color:var(--text-secondary)}
.table-view{margin-top:12px}
.table-view summary{cursor:pointer;font-size:12.5px;color:var(--link);width:max-content}
.table-view table{margin-top:8px}

#tip{position:fixed;z-index:10;max-width:340px;padding:8px 10px;border-radius:6px;
  background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
  box-shadow:0 4px 16px var(--shadow);font-size:12.5px}
.tip-title{font-weight:600;margin-bottom:4px}
.tip-row{display:flex;align-items:baseline;gap:6px}
.tip-row strong{white-space:nowrap}
.tip-row span{color:var(--text-secondary)}
.tip-key{width:12px;height:3px;border-radius:2px;flex:none;align-self:center}
footer{margin-top:60px;padding-top:14px;border-top:1px solid var(--axis);color:var(--text-secondary);font-size:12.5px;
  display:grid;gap:4px;max-width:96ch}

.back{display:inline-block;margin-bottom:18px;font-size:13px}
.scroll:focus-visible{outline-offset:4px}

/* uma seção por vez (?secao=drift), para apresentar ou capturar */
:root[data-solo] .rail,:root[data-solo] .main>:not(.solo){display:none}
:root[data-solo] .layout{display:block}
:root[data-solo] .main{padding-top:20px}
:root[data-solo] .block.solo{margin-top:0}

@media (max-width:980px){
  .layout{display:block}
  .rail{position:static;height:auto;border-right:0;border-bottom:1px solid var(--border);gap:16px;padding:20px}
  .run{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0 16px}
  .toc ul{display:flex;flex-wrap:wrap;gap:2px 16px}
  .toc a{border-left:0;padding:4px 0}
  .theme{margin-top:0}
  h1{font-size:28px}
  .board{table-layout:auto}.board col.signal{width:auto}
}
@media print{.rail,.filters,.table-view summary{display:none}.layout{display:block}.block,.stat{break-inside:avoid}}
""".replace("%DARK%", DARK)

JS = """
(function () {
  var root = document.documentElement;
  var params = new URLSearchParams(window.location.search);

  // Tema: segue o sistema; ?theme=light|dark na URL ou o botão do trilho forçam um modo.
  var asked = params.get('theme');
  if (asked === 'light' || asked === 'dark') { root.setAttribute('data-theme', asked); }
  var themeButton = document.getElementById('theme');
  function currentTheme() {
    return root.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function labelTheme() {
    themeButton.textContent = currentTheme() === 'dark' ? 'Usar tema claro' : 'Usar tema escuro';
  }
  themeButton.addEventListener('click', function () {
    root.setAttribute('data-theme', currentTheme() === 'dark' ? 'light' : 'dark');
    labelTheme();
  });
  labelTheme();

  // ?secao=drift mostra só aquela seção, para apresentar ou capturar a tela.
  var solo = params.get('secao');
  var soloTarget = solo ? document.getElementById(solo) : null;
  if (soloTarget) {
    soloTarget.classList.add('solo');
    root.setAttribute('data-solo', solo);
    var back = document.createElement('a');
    back.className = 'back solo';
    back.href = window.location.pathname + (asked ? '?theme=' + asked : '');
    back.textContent = 'Ver o boletim completo';
    soloTarget.parentNode.insertBefore(back, soloTarget);
  }

  var tip = document.getElementById('tip');
  function fill(data) {
    while (tip.firstChild) { tip.removeChild(tip.firstChild); }
    if (data.t) {
      var title = document.createElement('div');
      title.className = 'tip-title';
      title.textContent = data.t;
      tip.appendChild(title);
    }
    (data.r || []).forEach(function (row) {
      var line = document.createElement('div');
      line.className = 'tip-row';
      if (row.c) {
        var key = document.createElement('i');
        key.className = 'tip-key';
        key.style.background = row.c;
        line.appendChild(key);
      }
      var value = document.createElement('strong');
      value.textContent = row.v;
      var name = document.createElement('span');
      name.textContent = row.n;
      line.appendChild(value);
      line.appendChild(name);
      tip.appendChild(line);
    });
  }
  // O tooltip fica acima do ponto indicado (ou abaixo, se não couber), pode ser alcançado
  // pelo ponteiro sem sumir e fecha com Esc.
  var hideTimer = null;
  function place(x, top, bottom) {
    window.clearTimeout(hideTimer);
    tip.hidden = false;
    var left = Math.min(Math.max(8, x - tip.offsetWidth / 2), window.innerWidth - tip.offsetWidth - 8);
    var above = top - tip.offsetHeight - 8;
    tip.style.left = left + 'px';
    tip.style.top = (above < 8 ? (bottom === undefined ? top : bottom) + 8 : above) + 'px';
  }
  function hide() { window.clearTimeout(hideTimer); tip.hidden = true; }
  function hideSoon() { window.clearTimeout(hideTimer); hideTimer = window.setTimeout(hide, 200); }
  tip.addEventListener('pointerenter', function () { window.clearTimeout(hideTimer); });
  tip.addEventListener('pointerleave', hideSoon);
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape') { hide(); } });

  document.querySelectorAll('[data-tip]').forEach(function (mark) {
    var data = JSON.parse(mark.getAttribute('data-tip'));
    function show() {
      var box = mark.getBoundingClientRect();
      fill(data);
      place(box.left + box.width / 2, box.top, box.bottom);
    }
    mark.addEventListener('pointerenter', show);
    mark.addEventListener('pointerleave', hideSoon);
    mark.addEventListener('focus', show);
    mark.addEventListener('blur', hide);
  });

  // Grades (quadro de sinais e mapa de PSI): uma única parada de Tab; as setas andam entre as células.
  document.querySelectorAll('table[data-grid]').forEach(function (table) {
    var rows = Array.prototype.map.call(table.tBodies[0].rows, function (row) {
      return Array.prototype.slice.call(row.querySelectorAll('td'));
    });
    rows.forEach(function (cells, r) {
      cells.forEach(function (cell, c) {
        cell.setAttribute('tabindex', r === 0 && c === 0 ? '0' : '-1');
        cell.addEventListener('keydown', function (event) {
          var step = {ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0]}[event.key];
          var next = step && rows[r + step[0]] && rows[r + step[0]][c + step[1]];
          if (!next) { return; }
          event.preventDefault();
          cell.setAttribute('tabindex', '-1');
          next.setAttribute('tabindex', '0');
          next.focus();
        });
      });
    });
  });

  document.querySelectorAll('svg[data-points]').forEach(function (svg) {
    var points = JSON.parse(svg.getAttribute('data-points'));
    var cross = svg.querySelector('.crosshair');
    var width = svg.viewBox.baseVal.width;
    var current = points.length - 1;
    function show(index, y) {
      var box = svg.getBoundingClientRect();
      var point = points[index];
      current = index;
      cross.setAttribute('x1', point.x);
      cross.setAttribute('x2', point.x);
      cross.style.visibility = 'visible';
      fill(point);
      place(box.left + point.x * box.width / width, y === undefined ? box.top + box.height / 3 : y - 6);
    }
    function leave() { cross.style.visibility = 'hidden'; hideSoon(); }
    svg.addEventListener('pointermove', function (event) {
      var box = svg.getBoundingClientRect();
      var x = (event.clientX - box.left) * width / box.width;
      var nearest = 0;
      points.forEach(function (point, index) {
        if (Math.abs(point.x - x) < Math.abs(points[nearest].x - x)) { nearest = index; }
      });
      show(nearest, event.clientY);
    });
    svg.addEventListener('pointerleave', leave);
    svg.addEventListener('focus', function () { show(current); });
    svg.addEventListener('blur', function () { cross.style.visibility = 'hidden'; hide(); });
    svg.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') { show(Math.max(0, current - 1)); event.preventDefault(); }
      if (event.key === 'ArrowRight') { show(Math.min(points.length - 1, current + 1)); event.preventDefault(); }
    });
  });

  var buttons = document.querySelectorAll('[data-filter]');
  var counter = document.getElementById('alert-count');
  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      var severity = button.getAttribute('data-filter');
      var shown = 0;
      buttons.forEach(function (other) { other.setAttribute('aria-pressed', String(other === button)); });
      document.querySelectorAll('tr[data-sev]').forEach(function (row) {
        row.hidden = severity !== 'all' && row.getAttribute('data-sev') !== severity;
        if (!row.hidden) { shown += 1; }
      });
      if (counter) { counter.textContent = shown + (shown === 1 ? ' alerta exibido' : ' alertas exibidos'); }
    });
  });
  // Tabelas que não cabem na largura ganham foco de teclado para poderem ser roladas.
  function markScrollable() {
    document.querySelectorAll('.scroll').forEach(function (box) {
      if (box.scrollWidth > box.clientWidth + 1 && !box.querySelector('[data-grid]')) {
        box.setAttribute('tabindex', '0');
        box.setAttribute('role', 'region');
        box.setAttribute('aria-label', 'Tabela com rolagem horizontal');
      } else {
        box.removeAttribute('tabindex');
        box.removeAttribute('role');
        box.removeAttribute('aria-label');
      }
    });
  }
  markScrollable();
  window.addEventListener('resize', markScrollable);

  // Índice do trilho: marca a seção que está na parte de cima da janela.
  var tocLinks = {};
  document.querySelectorAll('.toc a').forEach(function (link) { tocLinks[link.getAttribute('href').slice(1)] = link; });
  if ('IntersectionObserver' in window) {
    var inView = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { inView[entry.target.id] = entry.isIntersecting; });
      var ids = Object.keys(tocLinks);
      var current = ids.filter(function (id) { return inView[id]; })[0];
      ids.forEach(function (id) {
        if (id === current) { tocLinks[id].setAttribute('aria-current', 'true'); } else { tocLinks[id].removeAttribute('aria-current'); }
      });
    }, {rootMargin: '-8% 0px -62% 0px'});
    Object.keys(tocLinks).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) { observer.observe(section); }
    });
  }
  root.setAttribute('data-ready', 'true');   // o script chegou ao fim sem erro
})();
"""
