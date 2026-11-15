"""Estilo e script embutidos no dashboard (ver ``src.dashboard``).

Cores por papel, com valores próprios para os modos claro e escuro: três séries
categóricas (validadas para daltonismo), uma rampa sequencial de um só matiz para
o PSI, o par divergente para efeitos positivos e negativos e as cores de status,
que sempre aparecem com ícone e rótulo.
"""

DARK = """color-scheme:dark;
  --page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--neg:#e66767;--pos:#3987e5;
  --seq-1:#0d366b;--seq-2:#184f95;--seq-3:#256abf;--seq-4:#5598e7;--seq-5:#9ec5f4;
  --seq-1-ink:#fff;--seq-2-ink:#fff;--seq-3-ink:#fff;--seq-4-ink:#0b0b0b;--seq-5-ink:#0b0b0b;--track:#0d366b"""

CSS = """
.viz-root{color-scheme:light;
  --page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--neg:#e34948;--pos:#2a78d6;
  --seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#5598e7;--seq-4:#256abf;--seq-5:#104281;
  --seq-1-ink:#0b0b0b;--seq-2-ink:#0b0b0b;--seq-3-ink:#0b0b0b;--seq-4-ink:#fff;--seq-5-ink:#fff;
  --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--track:#cde2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .viz-root{%DARK%}}
:root[data-theme="dark"] .viz-root{%DARK%}
*{box-sizing:border-box}
.viz-root{margin:0;padding:28px 16px 56px;background:var(--page);color:var(--text-primary);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;display:grid;gap:14px}
h1{font-size:26px;line-height:1.2;margin:2px 0 6px}
h2{font-size:17px;margin:0}
h3{font-size:13px;font-weight:600;margin:18px 0 6px;color:var(--text-secondary)}
p{margin:0}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text-secondary)}
a{color:inherit}
header{position:relative}
.theme{position:absolute;top:0;right:0;font:inherit;font-size:12px;padding:3px 10px;border-radius:14px;
  border:1px solid var(--axis);background:transparent;color:var(--text-secondary);cursor:pointer}
.eyebrow{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--text-secondary)}
.meta,.synthetic{color:var(--text-secondary);font-size:13px}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:13px}
.toc a{color:var(--text-secondary)}
.card{scroll-margin-top:12px}
.verdict{display:flex;gap:10px;align-items:baseline;margin:12px 0 8px;padding:10px 14px;
  background:var(--surface-1);border:1px solid var(--border);border-radius:10px;font-size:15px}
.verdict .chip{font-weight:600;white-space:nowrap}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:18px 20px;min-width:0}
.sub{color:var(--text-secondary);font-size:13px;margin-top:2px}
.reading{margin-top:12px;max-width:95ch}
.note{color:var(--text-secondary);font-size:12.5px;margin-top:10px;max-width:110ch}
.muted{color:var(--muted)}.strong{font-weight:600}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(176px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-label{font-size:12.5px;color:var(--text-secondary)}
.tile-value{font-size:28px;font-weight:600;line-height:1.25}
.tile-note{font-size:12px;color:var(--text-secondary)}
.spark{display:block;width:120px;height:30px;margin-top:6px}
.spark path{fill:none;stroke:var(--axis);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.spark circle{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:8px 28px}
.grid2>div{min-width:0}
.chip{display:inline-flex;align-items:baseline;gap:5px;white-space:nowrap}
.ico{font-size:12px;font-weight:700}
.ico-good{color:var(--good)}.ico-warning{color:var(--warning)}.ico-serious{color:var(--serious)}
.ico-critical{color:var(--critical)}.ico-neutral{color:var(--muted)}
.chart{display:block;width:100%;height:auto;overflow:visible}
.chart:focus-visible,[data-tip]:focus-visible{outline:2px solid var(--series-1);outline-offset:2px}
.chart text{font-family:inherit}
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
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;vertical-align:top}
thead th{font-size:12px;font-weight:600;color:var(--text-secondary);border-bottom:1px solid var(--axis);white-space:nowrap}
.data tbody tr{border-bottom:1px solid var(--grid)}
.data td{font-variant-numeric:tabular-nums}
.heat tbody th code{display:block;font-size:11px;color:var(--muted)}
.num{font-variant-numeric:tabular-nums}
.health td{white-space:nowrap}.health .chip{margin-right:2px}
.meter,.inline-bar{display:inline-block;vertical-align:middle;height:8px;border-radius:4px;margin-left:8px}
.meter{width:90px;background:var(--track)}
.meter span{display:block;height:100%;border-radius:4px;background:var(--series-1)}
.inline-bar{width:64px}
.causal td{white-space:nowrap}.causal td:first-child{white-space:normal}
.inline-bar span{display:block;height:100%;border-radius:0 4px 4px 0;background:var(--pos)}
.inline-bar.neg span{background:var(--neg);margin-left:auto;border-radius:4px 0 0 4px}
.heat{border-collapse:separate;border-spacing:2px;margin-top:12px}
.heat thead th{border:0;text-align:center;padding:4px 6px}
.heat thead th:first-child{text-align:left}
.heat thead a{display:block;font-weight:400;font-size:11px;color:var(--text-secondary)}
.heat tbody th{font-weight:500;padding:6px 10px 6px 0;min-width:230px}
.heat td{text-align:center;padding:9px 6px;border-radius:3px;font-variant-numeric:tabular-nums;min-width:70px;cursor:default}
.heat td span{margin-right:4px;font-size:10px}
.heat td:hover{filter:brightness(1.08)}
.heat .score-row th,.heat .score-row td{border-top:8px solid var(--surface-1)}
.tag{margin-left:8px;padding:1px 6px;border:1px solid var(--axis);border-radius:9px;font-size:10.5px;color:var(--text-secondary);white-space:nowrap}
.seq-1{background:var(--seq-1);color:var(--seq-1-ink)}.seq-2{background:var(--seq-2);color:var(--seq-2-ink)}
.seq-3{background:var(--seq-3);color:var(--seq-3-ink)}.seq-4{background:var(--seq-4);color:var(--seq-4-ink)}
.seq-5{background:var(--seq-5);color:var(--seq-5-ink)}
.scale{margin-top:10px}
.pill{display:inline-block;min-width:64px;padding:2px 8px;border-radius:3px;text-align:center}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 8px}
.filters button{font:inherit;font-size:13px;padding:4px 12px;border-radius:14px;border:1px solid var(--axis);
  background:transparent;color:var(--text-primary);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--text-primary);color:var(--surface-1);border-color:var(--text-primary)}
.alerts td:nth-child(4){min-width:320px}
.defs td:first-child{font-weight:600;white-space:nowrap}
.table-view{margin-top:10px}
.table-view summary{cursor:pointer;font-size:12.5px;color:var(--text-secondary)}
.table-view table{margin-top:8px}
#tip{position:fixed;z-index:10;pointer-events:none;max-width:320px;padding:8px 10px;border-radius:8px;
  background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
  box-shadow:0 4px 14px rgba(0,0,0,.18);font-size:12.5px}
.tip-title{color:var(--text-secondary);margin-bottom:4px}
.tip-row{display:flex;align-items:center;gap:6px;white-space:nowrap}
.tip-row span{color:var(--text-secondary)}
.tip-key{width:12px;height:3px;border-radius:2px;flex:none}
footer{color:var(--text-secondary);font-size:12px;display:grid;gap:4px;padding:4px 2px}
@media print{.filters,.theme,.table-view summary{display:none}.card,.tile{break-inside:avoid}}
""".replace("%DARK%", DARK)

JS = """
(function () {
  // Tema: segue o sistema; ?theme=light|dark na URL ou o botão do cabeçalho forçam um modo.
  var root = document.documentElement;
  var asked = new URLSearchParams(window.location.search).get('theme');
  if (asked === 'light' || asked === 'dark') { root.setAttribute('data-theme', asked); }
  document.getElementById('theme').addEventListener('click', function () {
    var current = root.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    root.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
  });

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
  function place(x, y) {
    tip.hidden = false;
    var left = Math.min(Math.max(8, x + 14), window.innerWidth - tip.offsetWidth - 8);
    var top = y - tip.offsetHeight - 12;
    tip.style.left = left + 'px';
    tip.style.top = (top < 8 ? y + 18 : top) + 'px';
  }
  function hide() { tip.hidden = true; }

  document.querySelectorAll('[data-tip]').forEach(function (mark) {
    var data = JSON.parse(mark.getAttribute('data-tip'));
    mark.addEventListener('pointermove', function (event) { fill(data); place(event.clientX, event.clientY); });
    mark.addEventListener('pointerleave', hide);
    mark.addEventListener('focus', function () {
      var box = mark.getBoundingClientRect();
      fill(data);
      place(box.left + box.width / 2, box.top);
    });
    mark.addEventListener('blur', hide);
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
      place(box.left + point.x * box.width / width, y === undefined ? box.top + box.height / 3 : y);
    }
    function leave() { cross.style.visibility = 'hidden'; hide(); }
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
    svg.addEventListener('blur', leave);
    svg.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') { show(Math.max(0, current - 1)); event.preventDefault(); }
      if (event.key === 'ArrowRight') { show(Math.min(points.length - 1, current + 1)); event.preventDefault(); }
    });
  });

  var buttons = document.querySelectorAll('[data-filter]');
  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      var severity = button.getAttribute('data-filter');
      buttons.forEach(function (other) { other.setAttribute('aria-pressed', String(other === button)); });
      document.querySelectorAll('tr[data-sev]').forEach(function (row) {
        row.hidden = severity !== 'all' && row.getAttribute('data-sev') !== severity;
      });
    });
  });
  root.setAttribute('data-ready', 'true');   // o script chegou ao fim sem erro
})();
"""
